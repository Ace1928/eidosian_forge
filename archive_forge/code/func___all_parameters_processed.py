def __all_parameters_processed(self):
    """
        Finish the argument processing.

        Should be called after all the web service operation's parameters have
        been successfully processed and, afterwards, no further parameter
        processing is allowed.

        Returns a 2-tuple containing the number of required & allowed
        arguments.

        See the _ArgParser class description for more detailed information.

        """
    assert self.active()
    sentinel_frame = self.__stack[0]
    self.__pop_frames_above(sentinel_frame)
    assert len(self.__stack) == 1
    self.__pop_top_frame()
    assert not self.active()
    args_required = sentinel_frame.args_required()
    args_allowed = sentinel_frame.args_allowed()
    self.__check_for_extra_arguments(args_required, args_allowed)
    return (args_required, args_allowed)