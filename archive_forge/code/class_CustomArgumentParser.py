import argparse
class CustomArgumentParser(argparse.ArgumentParser):
    """
    Custom argument parser that allows for the use of `-` or `_` in arguments passed and overrides the help for each
    when applicable.
    """

    def add_argument(self, *args, **kwargs):
        if 'action' in kwargs:
            if kwargs['action'] == 'store_true':
                kwargs['action'] = _StoreTrueAction
        else:
            kwargs['action'] = _StoreAction
        super().add_argument(*args, **kwargs)

    def add_argument_group(self, *args, **kwargs):
        group = CustomArgumentGroup(self, *args, **kwargs)
        self._action_groups.append(group)
        return group