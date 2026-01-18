from ..libmp.backend import xrange
import warnings
def __get_element(self, key):
    """
        Fast extraction of the i,j element from the matrix
            This function is for private use only because is unsafe:
                1. Does not check on the value of key it expects key to be a integer tuple (i,j)
                2. Does not check bounds
        """
    if key in self.__data:
        return self.__data[key]
    else:
        return self.ctx.zero