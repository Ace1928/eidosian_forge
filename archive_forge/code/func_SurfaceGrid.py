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
def SurfaceGrid(geometry, material, **kwargs):
    """A grid covering a surface.

    This will draw a line mesh overlaying the SurfaceGeometry.
    """
    nx = geometry.width_segments + 1
    ny = geometry.height_segments + 1
    vertices = geometry.attributes['position'].array
    lines = []
    for x in range(nx):
        g = Geometry(vertices=[vertices[y * nx + x, :].tolist() for y in range(ny)])
        lines.append(Line(g, material))
    for y in range(ny):
        g = Geometry(vertices=[vertices[y * nx + x, :].tolist() for x in range(nx)])
        lines.append(Line(g, material))

    def _update_lines(change):
        vertices = geometry.attributes['position'].array
        for x in range(nx):
            g = lines[x].geometry
            g.vertices = [vertices[y * nx + x, :].tolist() for y in range(ny)]
        for y in range(ny):
            g = lines[nx + y].geometry
            g.vertices = [vertices[y * nx + x, :].tolist() for x in range(nx)]
    geometry.attributes['position'].observe(_update_lines, names='array')
    return Group(children=lines, **kwargs)