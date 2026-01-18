def __push_frames(self, ancestry):
    """
        Push new frames representing given ancestry items.

        May only be given ancestry items other than None. Ancestry item None
        represents the internal sentinel item and should never appear in a
        given parameter's ancestry information.

        """
    for x in ancestry:
        assert x is not None
        self.__push_frame(x)