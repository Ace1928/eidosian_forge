import numpy as np
from onnx.reference.op_run import OpRun
def _common_run_shape(self, *args):
    num_loop_state_vars = len(args) - self.num_scan_inputs
    num_scan_outputs = len(args) - num_loop_state_vars
    output_directions = [0 if self.scan_output_directions is None or i >= len(self.scan_output_directions) else self.scan_output_directions[i] for i in range(num_scan_outputs)]
    max_dir_out = max(output_directions)
    if max_dir_out != 0:
        raise RuntimeError('Scan is not implemented for other output output_direction than 0.')
    output_axes = [0 if self.scan_output_axes is None or i >= len(self.scan_output_axes) else self.scan_output_axes[i] for i in range(num_scan_outputs)]
    max_axe_out = max(output_axes)
    if max_axe_out != 0:
        raise RuntimeError('Scan is not implemented for other output axes than 0.')
    state_names_in = self.input_names[:self.num_scan_inputs]
    state_names_out = self.output_names[:len(state_names_in)]
    scan_names_in = self.input_names[num_loop_state_vars:]
    scan_names_out = self.output_names[num_loop_state_vars:]
    scan_values = args[num_loop_state_vars:]
    states = args[:num_loop_state_vars]
    return (num_loop_state_vars, num_scan_outputs, output_directions, max_dir_out, output_axes, max_axe_out, state_names_in, state_names_out, scan_names_in, scan_names_out, scan_values, states)