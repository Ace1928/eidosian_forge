from oslo_config import cfg
@staticmethod
def _path_in_stack(stack):
    path = []
    for parent_res_name, stack_name in stack.path_in_stack():
        if parent_res_name is not None:
            path.append(','.join([parent_res_name, stack_name]))
        else:
            path.append(stack_name)
    return path