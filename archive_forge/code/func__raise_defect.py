from __future__ import print_function   # This version of olefile requires Python 2.7 or 3.5+.
import io
import sys
import struct, array, os.path, datetime, logging, warnings, traceback
def _raise_defect(self, defect_level, message, exception_type=OleFileError):
    """
        This method should be called for any defect found during file parsing.
        It may raise an OleFileError exception according to the minimal level chosen
        for the OleFileIO object.

        :param defect_level: defect level, possible values are:

            - DEFECT_UNSURE    : a case which looks weird, but not sure it's a defect
            - DEFECT_POTENTIAL : a potential defect
            - DEFECT_INCORRECT : an error according to specifications, but parsing can go on
            - DEFECT_FATAL     : an error which cannot be ignored, parsing is impossible

        :param message: string describing the defect, used with raised exception.
        :param exception_type: exception class to be raised, OleFileError by default
        """
    if defect_level >= self._raise_defects_level:
        log.error(message)
        raise exception_type(message)
    else:
        self.parsing_issues.append((exception_type, message))
        log.warning(message)