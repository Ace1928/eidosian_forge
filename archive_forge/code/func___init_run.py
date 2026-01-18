def __init_run(self, args, kwargs, extra_parameter_errors):
    """Initializes data for a new argument parsing run."""
    assert not self.active()
    self.__args = list(args)
    self.__kwargs = dict(kwargs)
    self.__extra_parameter_errors = extra_parameter_errors
    self.__args_count = len(args) + len(kwargs)
    self.__params_with_arguments = set()
    self.__stack = []
    self.__push_frame(None)