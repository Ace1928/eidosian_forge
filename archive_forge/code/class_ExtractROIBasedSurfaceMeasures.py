import os
from pathlib import Path
from nipype.interfaces.base import File, InputMultiPath, TraitedSpec, traits, isdefined
from nipype.interfaces.cat12.base import NestedCell, Cell
from nipype.interfaces.spm import SPMCommand
from nipype.interfaces.spm.base import SPMCommandInputSpec
from nipype.utils.filemanip import split_filename
class ExtractROIBasedSurfaceMeasures(SPMCommand):
    """
    Extract ROI-based surface values
    While ROI-based values for VBM (volume) data are automatically saved in the ``label`` folder as XML file it is
    necessary to additionally extract these values for surface data (except for thickness which is automatically
    extracted during segmentation). This has to be done after preprocessing the data and creating cortical surfaces.

    You can extract ROI-based values for cortical thickness but also for any other surface parameter that was extracted
    using the Extract Additional Surface Parameters such as volume, area, depth, gyrification and fractal dimension.


     http://www.neuro.uni-jena.de/cat12/CAT12-Manual.pdf#page=53

     Examples
     --------
    >>> # Template surface files
    >>> lh_atlas = 'lh.aparc_a2009s.freesurfer.annot'
    >>> rh_atlas = 'rh.aparc_a2009s.freesurfer.annot'
    >>> surf_files = ['lh.sphere.reg.structural.gii', 'rh.sphere.reg.structural.gii', 'lh.sphere.structural.gii', 'rh.sphere.structural.gii', 'lh.central.structural.gii', 'rh.central.structural.gii', 'lh.pbt.structural', 'rh.pbt.structural']
    >>> lh_measure = 'lh.area.structural'
    >>> extract_additional_measures = ExtractROIBasedSurfaceMeasures(surface_files=surf_files, lh_surface_measure=lh_measure, lh_roi_atlas=lh_atlas, rh_roi_atlas=rh_atlas)
    >>> extract_additional_measures.run() # doctest: +SKIP


    """
    input_spec = ExtractROIBasedSurfaceMeasuresInputSpec
    output_spec = ExtractROIBasedSurfaceMeasuresOutputSpec

    def __init__(self, **inputs):
        _local_version = SPMCommand().version
        if _local_version and '12.' in _local_version:
            self._jobtype = 'tools'
            self._jobname = 'cat.stools.surf2roi'
        SPMCommand.__init__(self, **inputs)

    def _format_arg(self, opt, spec, val):
        if opt == 'lh_surface_measure':
            return NestedCell(val)
        elif opt == 'lh_roi_atlas':
            return Cell2Str(val)
        return super(ExtractROIBasedSurfaceMeasures, self)._format_arg(opt, spec, val)

    def _list_outputs(self):
        outputs = self._outputs().get()
        pth, base, ext = split_filename(self.inputs.lh_surface_measure[0])
        outputs['label_files'] = [str(label) for label in Path(pth).glob('label/*') if label.is_file()]
        return outputs