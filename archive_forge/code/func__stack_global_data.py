from collections import OrderedDict
import importlib
def _stack_global_data(self, global_data_list_of_lists):
    global_data = list()
    for i in range(self._mpi_interface.size):
        global_data.extend(global_data_list_of_lists[i])
    return global_data