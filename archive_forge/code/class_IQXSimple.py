from typing import Dict, Any, Mapping
from qiskit.visualization.pulse_v2 import generators, layouts
class IQXSimple(dict):
    """Simple pulse stylesheet without channel notation.

    - Generate stepwise waveform envelope with latex pulse names.
    - Apply phase modulation to waveforms.
    - Do not show frame changes.
    - Show chart name.
    - Do not show snapshot and barrier.
    - Do not show acquire channels.
    - Channels are sorted by qubit index.
    """

    def __init__(self, **kwargs):
        super().__init__()
        style = {'formatter.general.fig_chart_height': 5, 'formatter.control.apply_phase_modulation': True, 'formatter.control.show_snapshot_channel': True, 'formatter.control.show_acquire_channel': False, 'formatter.control.show_empty_channel': False, 'formatter.control.auto_chart_scaling': False, 'formatter.control.axis_break': True, 'generator.waveform': [generators.gen_filled_waveform_stepwise, generators.gen_ibmq_latex_waveform_name], 'generator.frame': [], 'generator.chart': [generators.gen_chart_name, generators.gen_baseline], 'generator.snapshot': [], 'generator.barrier': [], 'layout.chart_channel_map': layouts.qubit_index_sort, 'layout.time_axis_map': layouts.time_map_in_ns, 'layout.figure_title': layouts.empty_title}
        style.update(**kwargs)
        self.update(style)

    def __repr__(self):
        return 'Simple pulse style sheet for publication.'