from typing import Dict, Any, Mapping
from qiskit.visualization.pulse_v2 import generators, layouts
class IQXDebugging(dict):
    """Pulse stylesheet for pulse programmers. Show details of instructions.

    # TODO: add more generators

    - Generate stepwise waveform envelope with latex pulse names.
    - Generate annotation for waveform height.
    - Apply phase modulation to waveforms.
    - Plot frame change symbol with raw operand values.
    - Show chart name and channel frequency.
    - Show snapshot and barrier.
    - Show acquire channels.
    - Channels are sorted by index and control channels are added to the end.
    """

    def __init__(self, **kwargs):
        super().__init__()
        style = {'formatter.control.apply_phase_modulation': True, 'formatter.control.show_snapshot_channel': True, 'formatter.control.show_acquire_channel': True, 'formatter.control.show_empty_channel': False, 'formatter.control.auto_chart_scaling': True, 'formatter.control.axis_break': True, 'generator.waveform': [generators.gen_filled_waveform_stepwise, generators.gen_ibmq_latex_waveform_name, generators.gen_waveform_max_value], 'generator.frame': [generators.gen_frame_symbol, generators.gen_raw_operand_values_compact], 'generator.chart': [generators.gen_chart_name, generators.gen_baseline, generators.gen_channel_freqs], 'generator.snapshot': [generators.gen_snapshot_symbol, generators.gen_snapshot_name], 'generator.barrier': [generators.gen_barrier], 'layout.chart_channel_map': layouts.channel_index_grouped_sort_u, 'layout.time_axis_map': layouts.time_map_in_ns, 'layout.figure_title': layouts.detail_title}
        style.update(**kwargs)
        self.update(style)

    def __repr__(self):
        return 'Pulse style sheet for pulse programmers.'