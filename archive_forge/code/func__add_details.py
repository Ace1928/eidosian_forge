import copy
def _add_details(self, info):
    for k, v in info.items():
        try:
            setattr(self, k, v)
            self._info[k] = v
        except AttributeError:
            pass