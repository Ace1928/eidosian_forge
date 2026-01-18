import re
import math
def __add_space_before_token(self):
    if self.space_before_token and (not self.just_added_newline()):
        if not self.non_breaking_space:
            self.set_wrap_point()
        self.current_line.push(' ')
    self.space_before_token = False