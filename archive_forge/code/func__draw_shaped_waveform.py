from __future__ import annotations
import re
from fractions import Fraction
from typing import Any
import numpy as np
from qiskit import pulse, circuit
from qiskit.pulse import instructions, library
from qiskit.visualization.exceptions import VisualizationError
from qiskit.visualization.pulse_v2 import drawings, types, device_info
def _draw_shaped_waveform(xdata: np.ndarray, ydata: np.ndarray, meta: dict[str, Any], channel: pulse.channels.PulseChannel, formatter: dict[str, Any]) -> list[drawings.LineData | drawings.BoxData | drawings.TextData]:
    """A private function that generates drawings of stepwise pulse lines.

    Args:
        xdata: Array of horizontal coordinate of waveform envelope.
        ydata: Array of vertical coordinate of waveform envelope.
        meta: Metadata dictionary of the waveform.
        channel: Channel associated with the waveform to draw.
        formatter: Dictionary of stylesheet settings.

    Returns:
        List of drawings.

    Raises:
        VisualizationError: When the waveform color for channel is not defined.
    """
    fill_objs: list[drawings.LineData | drawings.BoxData | drawings.TextData] = []
    resolution = formatter['general.vertical_resolution']
    xdata: np.ndarray = np.concatenate((xdata, [xdata[-1] + 1]))
    ydata = np.repeat(ydata, 2)
    re_y = np.real(ydata)
    im_y = np.imag(ydata)
    time: np.ndarray = np.concatenate(([xdata[0]], np.repeat(xdata[1:-1], 2), [xdata[-1]]))
    style = {'alpha': formatter['alpha.fill_waveform'], 'zorder': formatter['layer.fill_waveform'], 'linewidth': formatter['line_width.fill_waveform'], 'linestyle': formatter['line_style.fill_waveform']}
    try:
        color_real, color_imag = formatter['color.waveforms'][channel.prefix.upper()]
    except KeyError as ex:
        raise VisualizationError(f'Waveform color for channel type {channel.prefix} is not defined') from ex
    if np.any(re_y):
        re_valid_inds = _find_consecutive_index(re_y, resolution)
        re_style = {'color': color_real}
        re_style.update(style)
        re_meta = {'data': 'real'}
        re_meta.update(meta)
        re_xvals = time[re_valid_inds]
        re_yvals = re_y[re_valid_inds]
        real = drawings.LineData(data_type=types.WaveformType.REAL, channels=channel, xvals=re_xvals, yvals=re_yvals, fill=formatter['control.fill_waveform'], meta=re_meta, styles=re_style)
        fill_objs.append(real)
    if np.any(im_y):
        im_valid_inds = _find_consecutive_index(im_y, resolution)
        im_style = {'color': color_imag}
        im_style.update(style)
        im_meta = {'data': 'imag'}
        im_meta.update(meta)
        im_xvals = time[im_valid_inds]
        im_yvals = im_y[im_valid_inds]
        imag = drawings.LineData(data_type=types.WaveformType.IMAG, channels=channel, xvals=im_xvals, yvals=im_yvals, fill=formatter['control.fill_waveform'], meta=im_meta, styles=im_style)
        fill_objs.append(imag)
    return fill_objs