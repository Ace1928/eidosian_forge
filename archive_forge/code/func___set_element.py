from ..libmp.backend import xrange
import warnings
def __set_element(self, key, value):
    """
        Fast assignment of the i,j element in the matrix
            This function is unsafe:
                1. Does not check on the value of key it expects key to be a integer tuple (i,j)
                2. Does not check bounds
                3. Does not check the value type
                4. Does not reset the LU cache
        """
    if value:
        self.__data[key] = value
    elif key in self.__data:
        del self.__data[key]