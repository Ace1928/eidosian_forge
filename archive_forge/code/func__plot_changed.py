from ipywidgets import Widget, widget_serialization
from traitlets import Unicode, CInt, Instance, List, CFloat, Bool, observe, validate
import numpy as np
from ._package import npm_pkg_name
from ._version import EXTENSION_SPEC_VERSION
from .core.BufferAttribute import BufferAttribute
from .core.Geometry import Geometry
from .core.BufferGeometry import BufferGeometry
from .geometries.BoxGeometry_autogen import BoxGeometry
from .geometries.SphereGeometry_autogen import SphereGeometry
from .lights.AmbientLight_autogen import AmbientLight
from .lights.DirectionalLight_autogen import DirectionalLight
from .materials.Material_autogen import Material
from .materials.MeshLambertMaterial_autogen import MeshLambertMaterial
from .materials.SpriteMaterial_autogen import SpriteMaterial
from .objects.Group_autogen import Group
from .objects.Line_autogen import Line
from .objects.Mesh_autogen import Mesh
from .objects.Sprite_autogen import Sprite
from .textures.Texture_autogen import Texture
from .textures.DataTexture import DataTexture
from .textures.TextTexture_autogen import TextTexture
def _plot_changed(self, name, old, new):
    self.type = new.scenetree_json()['type']
    if self.type == 'object':
        self.type = new.scenetree_json()['geometry']['type']
        self.material = self.material_from_object(new)
    else:
        self.type = new.scenetree_json()['children'][0]['geometry']['type']
        self.material = self.material_from_other(new)
    if self.type == 'index_face_set':
        self.geometry = self.geometry_from_plot(new)
    elif self.type == 'sphere':
        self.geometry = self.geometry_from_sphere(new)
    elif self.type == 'box':
        self.geometry = self.geometry_from_box(new)