from saharaclient.api import parameters as params
def _get_node_processes(self, plugin):
    processes = []
    for proc_lst in plugin.node_processes.values():
        processes += proc_lst
    return [(proc_name, proc_name) for proc_name in processes]