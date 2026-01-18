import logging
from pyomo.common.log import is_debug_set
from pyomo.dataportal.factory import DataManagerFactory, UnknownDataManager
def _preprocess_options(self):
    """
        Preprocess the options for a data manager.
        """
    options = self._data_manager.options
    if options.data is None and (not options.set is None or not options.param is None or (not options.index is None)):
        options.data = []
        if not options.set is None:
            assert type(options.set) not in (list, tuple)
            options.data.append(options.set)
        if not options.index is None:
            options.data.append(options.index)
        if not options.param is None:
            if type(options.param) in (list, tuple):
                for item in options.param:
                    options.data.append(item)
            else:
                options.data.append(options.param)
    if options.data is None:
        return
    if type(options.data) in (list, tuple):
        ans = []
        for item in options.data:
            try:
                ans.append(item.local_name)
                self._model = item.model()
            except:
                ans.append(item)
        options.data = ans
    else:
        try:
            self._model = options.data.model()
            options.data = [self._data_manager.options.data.local_name]
        except:
            pass