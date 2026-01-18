def __push_frame(self, ancestry_item):
    """Push a new frame on top of the frame stack."""
    frame = self.__frame_factory(ancestry_item)
    self.__stack.append(frame)