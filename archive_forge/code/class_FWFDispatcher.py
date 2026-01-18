from typing import Optional, Sequence, Tuple, Union
from modin.core.io.text.text_file_dispatcher import TextFileDispatcher
class FWFDispatcher(TextFileDispatcher):
    """Class handles utils for reading of tables with fixed-width formatted lines."""

    @classmethod
    def check_parameters_support(cls, filepath_or_buffer, read_kwargs: dict, skiprows_md: Union[Sequence, callable, int], header_size: int) -> Tuple[bool, Optional[str]]:
        """
        Check support of parameters of `read_fwf` function.

        Parameters
        ----------
        filepath_or_buffer : str, path object or file-like object
            `filepath_or_buffer` parameter of `read_fwf` function.
        read_kwargs : dict
            Parameters of `read_fwf` function.
        skiprows_md : int, array or callable
            `skiprows` parameter modified for easier handling by Modin.
        header_size : int
            Number of rows that are used by header.

        Returns
        -------
        bool
            Whether passed parameters are supported or not.
        Optional[str]
            `None` if parameters are supported, otherwise an error
            message describing why parameters are not supported.
        """
        if read_kwargs['infer_nrows'] > 100:
            return (False, '`infer_nrows` is a significant portion of the number of rows, so Pandas may be faster')
        return super().check_parameters_support(filepath_or_buffer, read_kwargs, skiprows_md, header_size)