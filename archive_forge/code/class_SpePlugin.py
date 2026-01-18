from datetime import datetime
import logging
import os
from typing import (
import warnings
import numpy as np
from ..core.request import Request, IOMode, InitializationError
from ..core.v3_plugin_api import PluginV3, ImageProperties
class SpePlugin(PluginV3):

    def __init__(self, request: Request, check_filesize: bool=True, char_encoding: Optional[str]=None, sdt_meta: Optional[bool]=None) -> None:
        """Instantiate a new SPE file plugin object

        Parameters
        ----------
        request : Request
            A request object representing the resource to be operated on.
        check_filesize : bool
            If True, compute the number of frames from the filesize, compare it
            to the frame count in the file header, and raise a warning if the
            counts don't match. (Certain software may create files with
        char_encoding : str
            Deprecated. Exists for backwards compatibility; use ``char_encoding`` of
            ``metadata`` instead.
        sdt_meta : bool
            Deprecated. Exists for backwards compatibility; use ``sdt_control`` of
            ``metadata`` instead.

        """
        super().__init__(request)
        if request.mode.io_mode == IOMode.write:
            raise InitializationError('cannot write SPE files')
        if char_encoding is not None:
            warnings.warn('Passing `char_encoding` to the constructor is deprecated. Use `char_encoding` parameter of the `metadata()` method instead.', DeprecationWarning)
        self._char_encoding = char_encoding
        if sdt_meta is not None:
            warnings.warn('Passing `sdt_meta` to the constructor is deprecated. Use `sdt_control` parameter of the `metadata()` method instead.', DeprecationWarning)
        self._sdt_meta = sdt_meta
        self._file = self.request.get_file()
        try:
            info = self._parse_header(Spec.basic, 'latin1')
            self._file_header_ver = info['file_header_ver']
            self._dtype = Spec.dtypes[info['datatype']]
            self._shape = (info['ydim'], info['xdim'])
            self._len = info['NumFrames']
            if check_filesize:
                if info['file_header_ver'] >= 3:
                    data_end = info['xml_footer_offset']
                else:
                    self._file.seek(0, os.SEEK_END)
                    data_end = self._file.tell()
                line = data_end - Spec.data_start
                line //= self._shape[0] * self._shape[1] * self._dtype.itemsize
                if line != self._len:
                    warnings.warn(f'The file header of {self.request.filename} claims there are {self._len} frames, but there are actually {line} frames.')
                    self._len = min(line, self._len)
            self._file.seek(Spec.data_start)
        except Exception:
            raise InitializationError('SPE plugin cannot read the provided file.')

    def read(self, *, index: int=...) -> np.ndarray:
        """Read a frame or all frames from the file

        Parameters
        ----------
        index : int
            Select the index-th frame from the file. If index is `...`,
            select all frames and stack them along a new axis.

        Returns
        -------
        A Numpy array of pixel values.

        """
        if index is Ellipsis:
            read_offset = Spec.data_start
            count = self._shape[0] * self._shape[1] * self._len
            out_shape = (self._len, *self._shape)
        elif index < 0:
            raise IndexError(f'Index `{index}` is smaller than 0.')
        elif index >= self._len:
            raise IndexError(f'Index `{index}` exceeds the number of frames stored in this file (`{self._len}`).')
        else:
            read_offset = Spec.data_start + index * self._shape[0] * self._shape[1] * self._dtype.itemsize
            count = self._shape[0] * self._shape[1]
            out_shape = self._shape
        self._file.seek(read_offset)
        data = np.fromfile(self._file, dtype=self._dtype, count=count)
        return data.reshape(out_shape)

    def iter(self) -> Iterator[np.ndarray]:
        """Iterate over the frames in the file

        Yields
        ------
        A Numpy array of pixel values.
        """
        return (self.read(index=i) for i in range(self._len))

    def metadata(self, index: int=..., exclude_applied: bool=True, char_encoding: str='latin1', sdt_control: bool=True) -> Dict[str, Any]:
        """SPE specific metadata.

        Parameters
        ----------
        index : int
            Ignored as SPE files only store global metadata.
        exclude_applied : bool
            Ignored. Exists for API compatibility.
        char_encoding : str
            The encoding to use when parsing strings.
        sdt_control : bool
            If `True`, decode special metadata written by the
            SDT-control software if present.

        Returns
        -------
        metadata : dict
            Key-value pairs of metadata.

        Notes
        -----
        SPE v3 stores metadata as XML, whereas SPE v2 uses a binary format.

        .. rubric:: Supported SPE v2 Metadata fields

        ROIs : list of dict
            Regions of interest used for recording images. Each dict has the
            "top_left" key containing x and y coordinates of the top left corner,
            the "bottom_right" key with x and y coordinates of the bottom right
            corner, and the "bin" key with number of binned pixels in x and y
            directions.
        comments : list of str
            The SPE format allows for 5 comment strings of 80 characters each.
        controller_version : int
            Hardware version
        logic_output : int
            Definition of output BNC
        amp_hi_cap_low_noise : int
            Amp switching mode
        mode : int
            Timing mode
        exp_sec : float
            Alternative exposure in seconds
        date : str
            Date string
        detector_temp : float
            Detector temperature
        detector_type : int
            CCD / diode array type
        st_diode : int
            Trigger diode
        delay_time : float
            Used with async mode
        shutter_control : int
            Normal, disabled open, or disabled closed
        absorb_live : bool
            on / off
        absorb_mode : int
            Reference strip or file
        can_do_virtual_chip : bool
            True or False whether chip can do virtual chip
        threshold_min_live : bool
            on / off
        threshold_min_val : float
            Threshold minimum value
        threshold_max_live : bool
            on / off
        threshold_max_val : float
            Threshold maximum value
        time_local : str
            Experiment local time
        time_utc : str
            Experiment UTC time
        adc_offset : int
            ADC offset
        adc_rate : int
            ADC rate
        adc_type : int
            ADC type
        adc_resolution : int
            ADC resolution
        adc_bit_adjust : int
            ADC bit adjust
        gain : int
            gain
        sw_version : str
            Version of software which created this file
        spare_4 : bytes
            Reserved space
        readout_time : float
            Experiment readout time
        type : str
            Controller type
        clockspeed_us : float
            Vertical clock speed in microseconds
        readout_mode : ["full frame", "frame transfer", "kinetics", ""]
            Readout mode. Empty string means that this was not set by the
            Software.
        window_size : int
            Window size for Kinetics mode
        file_header_ver : float
            File header version
        chip_size : [int, int]
            x and y dimensions of the camera chip
        virt_chip_size : [int, int]
            Virtual chip x and y dimensions
        pre_pixels : [int, int]
            Pre pixels in x and y dimensions
        post_pixels : [int, int],
            Post pixels in x and y dimensions
        geometric : list of {"rotate", "reverse", "flip"}
            Geometric operations
        sdt_major_version : int
            (only for files created by SDT-control)
            Major version of SDT-control software
        sdt_minor_version : int
            (only for files created by SDT-control)
            Minor version of SDT-control software
        sdt_controller_name : str
            (only for files created by SDT-control)
            Controller name
        exposure_time : float
            (only for files created by SDT-control)
            Exposure time in seconds
        color_code : str
            (only for files created by SDT-control)
            Color channels used
        detection_channels : int
            (only for files created by SDT-control)
            Number of channels
        background_subtraction : bool
            (only for files created by SDT-control)
            Whether background subtraction war turned on
        em_active : bool
            (only for files created by SDT-control)
            Whether EM was turned on
        em_gain : int
            (only for files created by SDT-control)
            EM gain
        modulation_active : bool
            (only for files created by SDT-control)
            Whether laser modulation (“attenuate”) was turned on
        pixel_size : float
            (only for files created by SDT-control)
            Camera pixel size
        sequence_type : str
            (only for files created by SDT-control)
            Type of sequnce (standard, TOCCSL, arbitrary, …)
        grid : float
            (only for files created by SDT-control)
            Sequence time unit (“grid size”) in seconds
        n_macro : int
            (only for files created by SDT-control)
            Number of macro loops
        delay_macro : float
            (only for files created by SDT-control)
            Time between macro loops in seconds
        n_mini : int
            (only for files created by SDT-control)
            Number of mini loops
        delay_mini : float
            (only for files created by SDT-control)
            Time between mini loops in seconds
        n_micro : int (only for files created by SDT-control)
            Number of micro loops
        delay_micro : float (only for files created by SDT-control)
            Time between micro loops in seconds
        n_subpics : int
            (only for files created by SDT-control)
            Number of sub-pictures
        delay_shutter : float
            (only for files created by SDT-control)
            Camera shutter delay in seconds
        delay_prebleach : float
            (only for files created by SDT-control)
            Pre-bleach delay in seconds
        bleach_time : float
            (only for files created by SDT-control)
            Bleaching time in seconds
        recovery_time : float
            (only for files created by SDT-control)
            Recovery time in seconds
        comment : str
            (only for files created by SDT-control)
            User-entered comment. This replaces the "comments" field.
        datetime : datetime.datetime
            (only for files created by SDT-control)
            Combines the "date" and "time_local" keys. The latter two plus
            "time_utc" are removed.
        modulation_script : str
            (only for files created by SDT-control)
            Laser modulation script. Replaces the "spare_4" key.
        bleach_piezo_active : bool
            (only for files created by SDT-control)
            Whether piezo for bleaching was enabled
        """
        if self._file_header_ver < 3:
            if self._char_encoding is not None:
                char_encoding = self._char_encoding
            if self._sdt_meta is not None:
                sdt_control = self._sdt_meta
            return self._metadata_pre_v3(char_encoding, sdt_control)
        return self._metadata_post_v3()

    def _metadata_pre_v3(self, char_encoding: str, sdt_control: bool) -> Dict[str, Any]:
        """Extract metadata from SPE v2 files

        Parameters
        ----------
        char_encoding
            String character encoding
        sdt_control
            If `True`, try to decode special metadata written by the
            SDT-control software.

        Returns
        -------
        dict mapping metadata names to values.

        """
        m = self._parse_header(Spec.metadata, char_encoding)
        nr = m.pop('NumROI', None)
        nr = 1 if nr < 1 else nr
        m['ROIs'] = roi_array_to_dict(m['ROIs'][:nr])
        m['chip_size'] = [m.pop(k, None) for k in ('xDimDet', 'yDimDet')]
        m['virt_chip_size'] = [m.pop(k, None) for k in ('VChipXdim', 'VChipYdim')]
        m['pre_pixels'] = [m.pop(k, None) for k in ('XPrePixels', 'YPrePixels')]
        m['post_pixels'] = [m.pop(k, None) for k in ('XPostPixels', 'YPostPixels')]
        m['comments'] = [str(c) for c in m['comments']]
        g = []
        f = m.pop('geometric', 0)
        if f & 1:
            g.append('rotate')
        if f & 2:
            g.append('reverse')
        if f & 4:
            g.append('flip')
        m['geometric'] = g
        t = m['type']
        if 1 <= t <= len(Spec.controllers):
            m['type'] = Spec.controllers[t - 1]
        else:
            m['type'] = None
        r = m['readout_mode']
        if 1 <= r <= len(Spec.readout_modes):
            m['readout_mode'] = Spec.readout_modes[r - 1]
        else:
            m['readout_mode'] = None
        for k in ('absorb_live', 'can_do_virtual_chip', 'threshold_min_live', 'threshold_max_live'):
            m[k] = bool(m[k])
        if sdt_control:
            SDTControlSpec.extract_metadata(m, char_encoding)
        return m

    def _metadata_post_v3(self) -> Dict[str, Any]:
        """Extract XML metadata from SPE v3 files

        Returns
        -------
        dict with key `"__xml"`, whose value is the XML metadata
        """
        info = self._parse_header(Spec.basic, 'latin1')
        self._file.seek(info['xml_footer_offset'])
        xml = self._file.read()
        return {'__xml': xml}

    def properties(self, index: int=...) -> ImageProperties:
        """Standardized ndimage metadata.

        Parameters
        ----------
        index : int
            If the index is an integer, select the index-th frame and return
            its properties. If index is an Ellipsis (...), return the
            properties of all frames in the file stacked along a new batch
            dimension.

        Returns
        -------
        properties : ImageProperties
            A dataclass filled with standardized image metadata.
        """
        if index is Ellipsis:
            return ImageProperties(shape=(self._len, *self._shape), dtype=self._dtype, n_images=self._len, is_batch=True)
        return ImageProperties(shape=self._shape, dtype=self._dtype, is_batch=False)

    def _parse_header(self, spec: Mapping[str, Tuple], char_encoding: str) -> Dict[str, Any]:
        """Get information from SPE file header

        Parameters
        ----------
        spec
            Maps header entry name to its location, data type description and
            optionally number of entries. See :py:attr:`Spec.basic` and
            :py:attr:`Spec.metadata`.
        char_encoding
            String character encoding

        Returns
        -------
        Dict mapping header entry name to its value
        """
        ret = {}
        decode = np.vectorize(lambda x: x.decode(char_encoding))
        for name, sp in spec.items():
            self._file.seek(sp[0])
            cnt = 1 if len(sp) < 3 else sp[2]
            v = np.fromfile(self._file, dtype=sp[1], count=cnt)
            if v.dtype.kind == 'S' and name not in Spec.no_decode:
                try:
                    v = decode(v)
                except Exception:
                    warnings.warn(f'Failed to decode "{name}" metadata string. Check `char_encoding` parameter.')
            try:
                v = v.item()
            except ValueError:
                v = np.squeeze(v)
            ret[name] = v
        return ret