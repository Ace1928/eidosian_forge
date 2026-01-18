import base64
import numpy as np
from . import _marching_cubes_lewiner_luts as mcluts
from . import _marching_cubes_lewiner_cy
def _marching_cubes_lewiner(volume, level, spacing, gradient_direction, step_size, allow_degenerate, use_classic, mask):
    """Lewiner et al. algorithm for marching cubes. See
    marching_cubes_lewiner for documentation.

    """
    if not isinstance(volume, np.ndarray) or volume.ndim != 3:
        raise ValueError('Input volume should be a 3D numpy array.')
    if volume.shape[0] < 2 or volume.shape[1] < 2 or volume.shape[2] < 2:
        raise ValueError('Input array must be at least 2x2x2.')
    volume = np.ascontiguousarray(volume, np.float32)
    if level is None:
        level = 0.5 * (volume.min() + volume.max())
    else:
        level = float(level)
        if level < volume.min() or level > volume.max():
            raise ValueError('Surface level must be within volume data range.')
    if len(spacing) != 3:
        raise ValueError('`spacing` must consist of three floats.')
    step_size = int(step_size)
    if step_size < 1:
        raise ValueError('step_size must be at least one.')
    use_classic = bool(use_classic)
    L = _get_mc_luts()
    if mask is not None:
        if not mask.shape == volume.shape:
            raise ValueError('volume and mask must have the same shape.')
    func = _marching_cubes_lewiner_cy.marching_cubes
    vertices, faces, normals, values = func(volume, level, L, step_size, use_classic, mask)
    if not len(vertices):
        raise RuntimeError('No surface found at the given iso value.')
    vertices = np.fliplr(vertices)
    normals = np.fliplr(normals)
    faces.shape = (-1, 3)
    if gradient_direction == 'descent':
        faces = np.fliplr(faces)
    elif not gradient_direction == 'ascent':
        raise ValueError(f'Incorrect input {gradient_direction} in `gradient_direction`, see docstring.')
    if not np.array_equal(spacing, (1, 1, 1)):
        vertices = vertices * np.r_[spacing]
    if allow_degenerate:
        return (vertices, faces, normals, values)
    else:
        fun = _marching_cubes_lewiner_cy.remove_degenerate_faces
        return fun(vertices.astype(np.float32), faces, normals, values)