import os
class ProfileDatum(object):
    """Profile data point."""

    def __init__(self, device_name, node_exec_stats, file_path, line_number, func_name, op_type):
        """Constructor.

    Args:
      device_name: (string) name of the device.
      node_exec_stats: `NodeExecStats` proto.
      file_path: path to the source file involved in creating the op.
      line_number: line number in the file involved in creating the op.
      func_name: name of the function that the line belongs to.
      op_type: (string) Operation type.
    """
        self.device_name = device_name
        self.node_exec_stats = node_exec_stats
        self.file_path = file_path
        self.line_number = line_number
        self.func_name = func_name
        if self.file_path:
            self.file_line_func = '%s:%d(%s)' % (os.path.basename(self.file_path), self.line_number, self.func_name)
        else:
            self.file_line_func = ''
        self.op_type = op_type
        self.start_time = self.node_exec_stats.all_start_micros
        self.op_time = self.node_exec_stats.op_end_rel_micros - self.node_exec_stats.op_start_rel_micros

    @property
    def exec_time(self):
        """Op execution time plus pre- and post-processing."""
        return self.node_exec_stats.all_end_rel_micros