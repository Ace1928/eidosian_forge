import sys
from google.protobuf import message
from tensorflow.core.profiler import tfprof_options_pb2
from tensorflow.core.profiler import tfprof_output_pb2
from tensorflow.python.eager import context
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.profiler import option_builder
from tensorflow.python.profiler import tfprof_logger
from tensorflow.python.util import _pywrap_tfprof as print_mdl
from tensorflow.python.util.tf_export import tf_export
def _build_options(options):
    """Build tfprof.OptionsProto.

  Args:
    options: A dictionary of options.

  Returns:
    tfprof.OptionsProto.
  """
    opts = tfprof_options_pb2.OptionsProto()
    opts.max_depth = options.get('max_depth', 10)
    opts.min_bytes = options.get('min_bytes', 0)
    opts.min_peak_bytes = options.get('min_peak_bytes', 0)
    opts.min_residual_bytes = options.get('min_residual_bytes', 0)
    opts.min_output_bytes = options.get('min_output_bytes', 0)
    opts.min_micros = options.get('min_micros', 0)
    opts.min_accelerator_micros = options.get('min_accelerator_micros', 0)
    opts.min_cpu_micros = options.get('min_cpu_micros', 0)
    opts.min_params = options.get('min_params', 0)
    opts.min_float_ops = options.get('min_float_ops', 0)
    opts.min_occurrence = options.get('min_occurrence', 0)
    opts.step = options.get('step', -1)
    opts.order_by = options.get('order_by', 'name')
    for p in options.get('account_type_regexes', []):
        opts.account_type_regexes.append(p)
    for p in options.get('start_name_regexes', []):
        opts.start_name_regexes.append(p)
    for p in options.get('trim_name_regexes', []):
        opts.trim_name_regexes.append(p)
    for p in options.get('show_name_regexes', []):
        opts.show_name_regexes.append(p)
    for p in options.get('hide_name_regexes', []):
        opts.hide_name_regexes.append(p)
    opts.account_displayed_op_only = options.get('account_displayed_op_only', False)
    for p in options.get('select', []):
        opts.select.append(p)
    opts.output = options.get('output', 'stdout')
    opts.dump_to_file = options.get('dump_to_file', '')
    return opts