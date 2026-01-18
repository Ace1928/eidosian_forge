def __update_context(self, ancestry):
    if not ancestry:
        return
    match_result = self.__match_ancestry(ancestry)
    last_matching_frame, unmatched_ancestry = match_result
    self.__pop_frames_above(last_matching_frame)
    self.__push_frames(unmatched_ancestry)