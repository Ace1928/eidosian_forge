def __check_for_extra_arguments(self, args_required, args_allowed):
    """
        Report an error in case any extra arguments are detected.

        Does nothing if reporting extra arguments as exceptions has not been
        enabled.

        May only be called after the argument processing has been completed.

        """
    assert not self.active()
    if not self.__extra_parameter_errors:
        return
    if self.__kwargs:
        param_name = list(self.__kwargs.keys())[0]
        if param_name in self.__params_with_arguments:
            msg = "got multiple values for parameter '%s'"
        else:
            msg = "got an unexpected keyword argument '%s'"
        self.__error(msg % (param_name,))
    if self.__args:

        def plural_suffix(count):
            if count == 1:
                return ''
            return 's'

        def plural_was_were(count):
            if count == 1:
                return 'was'
            return 'were'
        expected = args_required
        if args_required != args_allowed:
            expected = '%d to %d' % (args_required, args_allowed)
        given = self.__args_count
        msg_parts = ['takes %s positional argument' % (expected,), plural_suffix(expected), ' but %d ' % (given,), plural_was_were(given), ' given']
        self.__error(''.join(msg_parts))