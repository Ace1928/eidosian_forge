import logging
from reportlab import rl_config
def add_generated_content(self, *C):
    self.__dict__.setdefault('_generated_content', []).extend(C)