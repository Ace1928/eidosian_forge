def __pop_top_frame(self):
    """Pops the top frame off the frame stack."""
    popped = self.__stack.pop()
    if self.__stack:
        self.__stack[-1].process_subframe(popped)