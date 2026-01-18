from datetime import datetime
import logging
import os
from typing import (
import warnings
import numpy as np
from ..core.request import Request, IOMode, InitializationError
from ..core.v3_plugin_api import PluginV3, ImageProperties
class SDTControlSpec:
    """Extract metadata written by the SDT-control software

    Some of it is encoded in the comment strings
    (see :py:meth:`parse_comments`). Also, date and time are encoded in a
    peculiar way (see :py:meth:`get_datetime`). Use :py:meth:`extract_metadata`
    to update the metadata dict.
    """
    months = {'Jän': 1, 'Jan': 1, 'Feb': 2, 'Mär': 3, 'Mar': 3, 'Apr': 4, 'Mai': 5, 'May': 5, 'Jun': 6, 'Jul': 7, 'Aug': 8, 'Sep': 9, 'Okt': 10, 'Oct': 10, 'Nov': 11, 'Dez': 12, 'Dec': 12}
    sequence_types = {'SEQU': 'standard', 'SETO': 'TOCCSL', 'KINE': 'kinetics', 'SEAR': 'arbitrary'}

    class CommentDesc:
        """Describe how to extract a metadata entry from a comment string"""
        n: int
        'Which of the 5 SPE comment fields to use.'
        slice: slice
        'Which characters from the `n`-th comment to use.'
        cvt: Callable[[str], Any]
        'How to convert characters to something useful.'
        scale: Union[None, float]
        'Optional scaling factor for numbers'

        def __init__(self, n: int, slice: slice, cvt: Callable[[str], Any]=str, scale: Optional[float]=None):
            self.n = n
            self.slice = slice
            self.cvt = cvt
            self.scale = scale
    comment_fields = {(5, 0): {'sdt_major_version': CommentDesc(4, slice(66, 68), int), 'sdt_minor_version': CommentDesc(4, slice(68, 70), int), 'sdt_controller_name': CommentDesc(4, slice(0, 6), str), 'exposure_time': CommentDesc(1, slice(64, 73), float, 10 ** (-6)), 'color_code': CommentDesc(4, slice(10, 14), str), 'detection_channels': CommentDesc(4, slice(15, 16), int), 'background_subtraction': CommentDesc(4, 14, lambda x: x == 'B'), 'em_active': CommentDesc(4, 32, lambda x: x == 'E'), 'em_gain': CommentDesc(4, slice(28, 32), int), 'modulation_active': CommentDesc(4, 33, lambda x: x == 'A'), 'pixel_size': CommentDesc(4, slice(25, 28), float, 0.1), 'sequence_type': CommentDesc(4, slice(6, 10), lambda x: __class__.sequence_types[x]), 'grid': CommentDesc(4, slice(16, 25), float, 10 ** (-6)), 'n_macro': CommentDesc(1, slice(0, 4), int), 'delay_macro': CommentDesc(1, slice(10, 19), float, 10 ** (-3)), 'n_mini': CommentDesc(1, slice(4, 7), int), 'delay_mini': CommentDesc(1, slice(19, 28), float, 10 ** (-6)), 'n_micro': CommentDesc(1, slice(7, 10), int), 'delay_micro': CommentDesc(1, slice(28, 37), float, 10 ** (-6)), 'n_subpics': CommentDesc(1, slice(7, 10), int), 'delay_shutter': CommentDesc(1, slice(73, 79), float, 10 ** (-6)), 'delay_prebleach': CommentDesc(1, slice(37, 46), float, 10 ** (-6)), 'bleach_time': CommentDesc(1, slice(46, 55), float, 10 ** (-6)), 'recovery_time': CommentDesc(1, slice(55, 64), float, 10 ** (-6))}, (5, 1): {'bleach_piezo_active': CommentDesc(4, slice(34, 35), lambda x: x == 'z')}}

    @staticmethod
    def get_comment_version(comments: Sequence[str]) -> Tuple[int, int]:
        """Get the version of SDT-control metadata encoded in the comments

        Parameters
        ----------
        comments
            List of SPE file comments, typically ``metadata["comments"]``.

        Returns
        -------
        Major and minor version. ``-1, -1`` if detection failed.
        """
        if comments[4][70:76] != 'COMVER':
            return (-1, -1)
        try:
            return (int(comments[4][76:78]), int(comments[4][78:80]))
        except ValueError:
            return (-1, -1)

    @staticmethod
    def parse_comments(comments: Sequence[str], version: Tuple[int, int]) -> Dict[str, Any]:
        """Extract SDT-control metadata from comments

        Parameters
        ----------
        comments
            List of SPE file comments, typically ``metadata["comments"]``.
        version
            Major and minor version of SDT-control metadata format

        Returns
        -------
        Dict of metadata
        """
        sdt_md = {}
        for minor in range(version[1] + 1):
            try:
                cmt = __class__.comment_fields[version[0], minor]
            except KeyError:
                continue
            for name, spec in cmt.items():
                try:
                    v = spec.cvt(comments[spec.n][spec.slice])
                    if spec.scale is not None:
                        v *= spec.scale
                    sdt_md[name] = v
                except Exception as e:
                    warnings.warn(f'Failed to decode SDT-control metadata field `{name}`: {e}')
                    sdt_md[name] = None
        if version not in __class__.comment_fields:
            supported_ver = ', '.join(map(lambda x: f'{x[0]}.{x[1]:02}', __class__.comment_fields))
            warnings.warn(f'Unsupported SDT-control metadata version {version[0]}.{version[1]:02}. Only versions {supported_ver} are supported. Some or all SDT-control metadata may be missing.')
        comment = comments[0] + comments[2]
        sdt_md['comment'] = comment.strip()
        return sdt_md

    @staticmethod
    def get_datetime(date: str, time: str) -> Union[datetime, None]:
        """Turn date and time saved by SDT-control into proper datetime object

        Parameters
        ----------
        date
            SPE file date, typically ``metadata["date"]``.
        time
            SPE file date, typically ``metadata["time_local"]``.

        Returns
        -------
        File's datetime if parsing was succsessful, else None.
        """
        try:
            month = __class__.months[date[2:5]]
            return datetime(int(date[5:9]), month, int(date[0:2]), int(time[0:2]), int(time[2:4]), int(time[4:6]))
        except Exception as e:
            logger.info(f'Failed to decode date from SDT-control metadata: {e}.')

    @staticmethod
    def extract_metadata(meta: Mapping, char_encoding: str='latin1'):
        """Extract SDT-control metadata from SPE metadata

        SDT-control stores some metadata in comments and other fields.
        Extract them and remove unused entries.

        Parameters
        ----------
        meta
            SPE file metadata. Modified in place.
        char_encoding
            Character encoding used to decode strings in the metadata.
        """
        comver = __class__.get_comment_version(meta['comments'])
        if any((c < 0 for c in comver)):
            logger.debug('SDT-control comments not found.')
            return
        sdt_meta = __class__.parse_comments(meta['comments'], comver)
        meta.pop('comments')
        meta.update(sdt_meta)
        dt = __class__.get_datetime(meta['date'], meta['time_local'])
        if dt:
            meta['datetime'] = dt
            meta.pop('date')
            meta.pop('time_local')
        sp4 = meta['spare_4']
        try:
            meta['modulation_script'] = sp4.decode(char_encoding)
            meta.pop('spare_4')
        except UnicodeDecodeError:
            warnings.warn('Failed to decode SDT-control laser modulation script. Bad char_encoding?')
        meta.pop('time_utc')
        meta.pop('exposure_sec')