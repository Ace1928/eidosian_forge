from __future__ import annotations
import base64
import json
import sys
import zipfile
from abc import abstractmethod
from typing import (
from urllib.request import urlopen
import numpy as np
import param
from bokeh.models import LinearColorMapper
from bokeh.util.serialization import make_globally_unique_id
from pyviz_comms import JupyterComm
from ...param import ParamMethod
from ...util import isfile, lazy_load
from ..base import PaneBase
from ..plot import Bokeh
from .enums import PRESET_CMAPS
class VTKVolume(AbstractVTK):
    """
    The `VTKVolume` pane renders 3d volumetric data defined on regular grids.
    It may be constructed from a 3D NumPy array or a vtkVolume.

    The pane provides a number of interactive control which can be set either
    through callbacks from Python or Javascript callbacks.

    Reference: https://panel.holoviz.org/reference/panes/VTKVolume.html

    :Example:

    >>> pn.extension('vtk')
    >>> VTKVolume(
    ...    data_matrix, spacing=(3,2,1), interpolation='nearest',
    ...    edge_gradient=0, sampling=0,
    ...    sizing_mode='stretch_width', height=400,
    ... )
    """
    ambient = param.Number(default=0.2, step=0.01, doc='\n        Value to control the ambient lighting. It is the light an\n        object gives even in the absence of strong light. It is\n        constant in all directions.')
    controller_expanded = param.Boolean(default=True, doc='\n        If True the volume controller panel options is expanded in the view')
    colormap = param.Selector(default='erdc_rainbow_bright', objects=PRESET_CMAPS, doc='\n        Name of the colormap used to transform pixel value in color.')
    diffuse = param.Number(default=0.7, step=0.01, doc='\n        Value to control the diffuse Lighting. It relies on both the\n        light direction and the object surface normal.')
    display_volume = param.Boolean(default=True, doc='\n        If set to True, the 3D representation of the volume is\n        displayed using ray casting.')
    display_slices = param.Boolean(default=False, doc='\n        If set to true, the orthgonal slices in the three (X, Y, Z)\n        directions are displayed. Position of each slice can be\n        controlled using slice_(i,j,k) parameters.')
    edge_gradient = param.Number(default=0.4, bounds=(0, 1), step=0.01, doc='\n        Parameter to adjust the opacity of the volume based on the\n        gradient between voxels.')
    interpolation = param.Selector(default='fast_linear', objects=['fast_linear', 'linear', 'nearest'], doc='\n        interpolation type for sampling a volume. `nearest`\n        interpolation will snap to the closest voxel, `linear` will\n        perform trilinear interpolation to compute a scalar value from\n        surrounding voxels.  `fast_linear` under WebGL 1 will perform\n        bilinear interpolation on X and Y but use nearest for Z. This\n        is slightly faster than full linear at the cost of no Z axis\n        linear interpolation.')
    mapper = param.Dict(doc='Lookup Table in format {low, high, palette}')
    max_data_size = param.Number(default=256 ** 3 * 2 / 1000000.0, doc='\n        Maximum data size transfer allowed without subsampling')
    nan_opacity = param.Number(default=1.0, bounds=(0.0, 1.0), doc='\n        Opacity applied to nan values in slices')
    origin = param.Tuple(default=None, length=3, allow_None=True)
    render_background = param.Color(default='#52576e', doc='\n        Allows to specify the background color of the 3D rendering.\n        The value must be specified as an hexadecimal color string.')
    rescale = param.Boolean(default=False, doc='\n        If set to True the colormap is rescaled between min and max\n        value of the non-transparent pixel, otherwise  the full range\n        of the pixel values are used.')
    shadow = param.Boolean(default=True, doc='\n        If set to False, then the mapper for the volume will not\n        perform shading computations, it is the same as setting\n        ambient=1, diffuse=0, specular=0.')
    sampling = param.Number(default=0.4, bounds=(0, 1), step=0.01, doc='\n        Parameter to adjust the distance between samples used for\n        rendering. The lower the value is the more precise is the\n        representation but it is more computationally intensive.')
    spacing = param.Tuple(default=(1, 1, 1), length=3, doc='\n        Distance between voxel in each direction')
    specular = param.Number(default=0.3, step=0.01, doc='\n        Value to control specular lighting. It is the light reflects\n        back toward the camera when hitting the object.')
    specular_power = param.Number(default=8.0, doc='\n        Specular power refers to how much light is reflected in a\n        mirror like fashion, rather than scattered randomly in a\n        diffuse manner.')
    slice_i = param.Integer(per_instance=True, doc='\n        Integer parameter to control the position of the slice normal\n        to the X direction.')
    slice_j = param.Integer(per_instance=True, doc='\n        Integer parameter to control the position of the slice normal\n        to the Y direction.')
    slice_k = param.Integer(per_instance=True, doc='\n        Integer parameter to control the position of the slice normal\n        to the Z direction.')
    _serializers = {}
    _rename: ClassVar[Mapping[str, str | None]] = {'max_data_size': None, 'spacing': None, 'origin': None}
    _updates = True

    def __init__(self, object=None, **params):
        super().__init__(object, **params)
        self._sub_spacing = self.spacing
        self._update(None, None)

    @classmethod
    def applies(cls, obj: Any) -> float | bool | None:
        if isinstance(obj, np.ndarray) and obj.ndim == 3 or any([isinstance(obj, k) for k in cls._serializers.keys()]):
            return True
        elif 'vtk' not in sys.modules and 'vtkmodules' not in sys.modules:
            return False
        else:
            import vtk
            return isinstance(obj, vtk.vtkImageData)

    def _get_model(self, doc: Document, root: Optional[Model]=None, parent: Optional[Model]=None, comm: Optional[Comm]=None) -> Model:
        VTKVolumePlot = lazy_load('panel.models.vtk', 'VTKVolumePlot', isinstance(comm, JupyterComm), root)
        props = self._process_param_change(self._init_params())
        if self._volume_data is not None:
            props['data'] = self._volume_data
        model = VTKVolumePlot(**props)
        if root is None:
            root = model
        self._link_props(model, ['colormap', 'orientation_widget', 'camera', 'mapper', 'controller_expanded', 'nan_opacity'], doc, root, comm)
        self._models[root.ref['id']] = (model, parent)
        return model

    def _update_object(self, ref, doc, root, parent, comm):
        self._legend = None
        super()._update_object(ref, doc, root, parent, comm)

    def _get_object_dimensions(self):
        if isinstance(self.object, np.ndarray):
            return self.object.shape
        else:
            return self.object.GetDimensions()

    def _process_param_change(self, msg):
        msg = super()._process_param_change(msg)
        if self.object is not None:
            slice_params = {'slice_i': 0, 'slice_j': 1, 'slice_k': 2}
            for k, v in msg.items():
                sub_dim = self._subsample_dimensions
                ori_dim = self._orginal_dimensions
                if k in slice_params:
                    index = slice_params[k]
                    msg[k] = int(np.round(v * sub_dim[index] / ori_dim[index]))
        return msg

    def _process_property_change(self, msg):
        msg = super()._process_property_change(msg)
        if self.object is not None:
            slice_params = {'slice_i': 0, 'slice_j': 1, 'slice_k': 2}
            for k, v in msg.items():
                sub_dim = self._subsample_dimensions
                ori_dim = self._orginal_dimensions
                if k in slice_params:
                    index = slice_params[k]
                    msg[k] = int(np.round(v * ori_dim[index] / sub_dim[index]))
        return msg

    def _update(self, ref: str, model: Model) -> None:
        self._volume_data = self._get_volume_data()
        if self._volume_data is not None:
            self._orginal_dimensions = self._get_object_dimensions()
            self._subsample_dimensions = self._volume_data['dims']
            self.param.slice_i.bounds = (0, self._orginal_dimensions[0] - 1)
            self.slice_i = (self._orginal_dimensions[0] - 1) // 2
            self.param.slice_j.bounds = (0, self._orginal_dimensions[1] - 1)
            self.slice_j = (self._orginal_dimensions[1] - 1) // 2
            self.param.slice_k.bounds = (0, self._orginal_dimensions[2] - 1)
            self.slice_k = (self._orginal_dimensions[2] - 1) // 2
        if model is not None:
            model.data = self._volume_data

    @classmethod
    def register_serializer(cls, class_type, serializer):
        """
        Register a serializer for a given type of class.
        A serializer is a function which take an instance of `class_type`
        (like a vtk.vtkImageData) as input and return a numpy array of the data
        """
        cls._serializers.update({class_type: serializer})

    def _volume_from_array(self, sub_array):
        return dict(buffer=base64encode(sub_array.ravel(order='F')), dims=sub_array.shape, spacing=self._sub_spacing, origin=self.origin, data_range=(np.nanmin(sub_array), np.nanmax(sub_array)), dtype=sub_array.dtype.name)

    def _get_volume_data(self):
        if self.object is None:
            return None
        elif isinstance(self.object, np.ndarray):
            return self._volume_from_array(self._subsample_array(self.object))
        else:
            available_serializer = [v for k, v in VTKVolume._serializers.items() if isinstance(self.object, k)]
            if not available_serializer:
                import vtk
                from vtk.util import numpy_support

                def volume_serializer(inst):
                    imageData = inst.object
                    array = numpy_support.vtk_to_numpy(imageData.GetPointData().GetScalars())
                    dims = imageData.GetDimensions()
                    inst.spacing = imageData.GetSpacing()
                    inst.origin = imageData.GetOrigin()
                    return inst._volume_from_array(inst._subsample_array(array.reshape(dims, order='F')))
                VTKVolume.register_serializer(vtk.vtkImageData, volume_serializer)
                serializer = volume_serializer
            else:
                serializer = available_serializer[0]
            return serializer(self)

    def _subsample_array(self, array):
        original_shape = array.shape
        spacing = self.spacing
        extent = tuple(((o_s - 1) * s for o_s, s in zip(original_shape, spacing)))
        dim_ratio = np.cbrt(array.nbytes / 1000000.0 / self.max_data_size)
        max_shape = tuple((int(o_s / dim_ratio) for o_s in original_shape))
        dowsnscale_factor = [max(o_s, m_s) / m_s for m_s, o_s in zip(max_shape, original_shape)]
        if any([d_f > 1 for d_f in dowsnscale_factor]):
            try:
                import scipy.ndimage as nd
                sub_array = nd.interpolation.zoom(array, zoom=[1 / d_f for d_f in dowsnscale_factor], order=0, mode='nearest')
            except ImportError:
                sub_array = array[::int(np.ceil(dowsnscale_factor[0])), ::int(np.ceil(dowsnscale_factor[1])), ::int(np.ceil(dowsnscale_factor[2]))]
            self._sub_spacing = tuple((e / (s - 1) for e, s in zip(extent, sub_array.shape)))
        else:
            sub_array = array
            self._sub_spacing = self.spacing
        return sub_array