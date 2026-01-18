import errno
import shlex
import subprocess
def _merge_args_opts(args_opts_dict, **kwargs):
    """Merge options with their corresponding arguments.

    Iterates over the dictionary holding arguments (keys) and options (values). Merges each
    options string with its corresponding argument.

    :param dict args_opts_dict: a dictionary of arguments and options
    :param dict kwargs: *input_option* - if specified prepends ``-i`` to input argument
    :return: merged list of strings with arguments and their corresponding options
    :rtype: list
    """
    merged = []
    if not args_opts_dict:
        return merged
    for arg, opt in args_opts_dict.items():
        if not _is_sequence(opt):
            opt = shlex.split(opt or '')
        merged += opt
        if not arg:
            continue
        if 'add_input_option' in kwargs:
            merged.append('-i')
        merged.append(arg)
    return merged