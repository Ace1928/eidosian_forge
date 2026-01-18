from pathlib import Path
import logging
def _update_shaders(path=None, file_changed=None):
    names = ['volr-fragment', 'volr-vertex', 'mesh-vertex', 'mesh-fragment', 'scatter-vertex', 'scatter-fragment', 'shadow-vertex', 'shadow-fragment']
    for figure in _figures:
        shaders = {}
        for name in names:
            shader_path = path / (name + '.glsl')
            with shader_path.open() as f:
                shaders[name] = f.read()
        figure._shaders = shaders