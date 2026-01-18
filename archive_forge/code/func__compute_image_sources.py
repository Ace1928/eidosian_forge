import math
from typing import Optional, Tuple, Union
import torch
import torchaudio
from torch import Tensor
def _compute_image_sources(room: torch.Tensor, source: torch.Tensor, max_order: int, absorption: torch.Tensor, scatter: Optional[torch.Tensor]=None) -> Tuple[Tensor, Tensor]:
    """Compute image sources in a shoebox-like room.

    Args:
        room (torch.Tensor): The 1D Tensor to determine the room size. The shape is
            `(D,)`, where ``D`` is 2 if room is a 2D room, or 3 if room is a 3D room.
        source (torch.Tensor): The coordinate of the sound source. Tensor with dimensions
            `(D)`.
        max_order (int): The maximum number of reflections of the source.
        absorption (torch.Tensor): The absorption coefficients of wall materials.
            ``absorption`` is a Tensor with dimensions `(num_band, num_wall)`.
            The shape options are ``[(1, 4), (1, 6), (7, 4), (7, 6)]``.
            ``num_band`` is `1` if the coefficients is the same for all frequencies, or is `7`
            if the coefficients are different to different frequencies. `7` refers to the default number
            of octave bands. (See note in `simulate_rir_ism` method).
            ``num_wall`` is `4` if the room is a 2D room, representing absorption coefficients
            of ``"west"``, ``"east"``, ``"south"``, and ``"north"`` walls, respectively.
            Or it is `6` if the room is a 3D room, representing absorption coefficients
            of ``"west"``, ``"east"``, ``"south"``, ``"north"``, ``"floor"``, and ``"ceiling"``, respectively.
        scatter (torch.Tensor): The scattering coefficients of wall materials.
            The shape of ``scatter`` must match that of ``absorption``. If ``None``, it is not
            used in image source computation. (Default: ``None``)

    Returns:
        (torch.Tensor): The coordinates of all image sources within ``max_order`` number of reflections.
            Tensor with dimensions `(num_image_source, D)`.
        (torch.Tensor): The attenuation of corresponding image sources. Tensor with dimensions
            `(num_band, num_image_source)`.
    """
    if scatter is None:
        tr = torch.sqrt(1 - absorption)
    else:
        tr = torch.sqrt(1 - absorption) * torch.sqrt(1 - scatter)
    ind = torch.arange(-max_order, max_order + 1, device=source.device)
    if room.shape[0] == 2:
        XYZ = torch.meshgrid(ind, ind, indexing='ij')
    else:
        XYZ = torch.meshgrid(ind, ind, ind, indexing='ij')
    XYZ = torch.stack([c.reshape((-1,)) for c in XYZ], dim=-1)
    XYZ = XYZ[XYZ.abs().sum(dim=-1) <= max_order]
    d = room[None, :]
    s = source[None, :]
    img_loc = torch.where(XYZ % 2 == 1, d * (XYZ + 1) - s, d * XYZ + s)
    exp_lo = abs(torch.floor(XYZ / 2))
    exp_hi = abs(torch.floor((XYZ + 1) / 2))
    t_lo = tr[:, ::2].unsqueeze(1).repeat(1, XYZ.shape[0], 1)
    t_hi = tr[:, 1::2].unsqueeze(1).repeat(1, XYZ.shape[0], 1)
    att = torch.prod(t_lo ** exp_lo * t_hi ** exp_hi, dim=-1)
    return (img_loc, att)