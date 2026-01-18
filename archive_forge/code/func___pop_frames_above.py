def __pop_frames_above(self, frame):
    """Pops all the frames above, but not including the given frame."""
    while self.__stack[-1] is not frame:
        self.__pop_top_frame()
    assert self.__stack