import os
import time
import warnings
import numpy as np
from .. import _HAS_FFMPEG
from ..utils import *
class VideoReaderAbstract(object):
    """Reads frames
    """
    INFO_AVERAGE_FRAMERATE = None
    INFO_WIDTH = None
    INFO_HEIGHT = None
    INFO_PIX_FMT = None
    INFO_DURATION = None
    INFO_NB_FRAMES = None
    DEFAULT_FRAMERATE = 25.0
    DEFAULT_INPUT_PIX_FMT = 'yuvj444p'
    OUTPUT_METHOD = None

    def __init__(self, filename, inputdict=None, outputdict=None, verbosity=0):
        """Initializes FFmpeg in reading mode with the given parameters

        During initialization, additional parameters about the video file
        are parsed using :func:`skvideo.io.ffprobe`. Then FFmpeg is launched
        as a subprocess. Parameters passed into inputdict are parsed and
        used to set as internal variables about the video. If the parameter,
        such as "Height" is not found in the inputdict, it is found through
        scanning the file's header information. If not in the header, ffprobe
        is used to decode the file to determine the information. In the case
        that the information is not supplied and connot be inferred from the
        input file, a ValueError exception is thrown.

        Parameters
        ----------
        filename : string
            Video file path

        inputdict : dict
            Input dictionary parameters, i.e. how to interpret the input file.

        outputdict : dict
            Output dictionary parameters, i.e. how to encode the data
            when sending back to the python process.

        Returns
        -------
        none

        """
        assert _HAS_FFMPEG, 'Cannot find installation of real FFmpeg (which comes with ffprobe).'
        self._filename = filename
        self.verbosity = verbosity
        if not inputdict:
            inputdict = {}
        if not outputdict:
            outputdict = {}
        _, self.extension = os.path.splitext(filename)
        self.size = os.path.getsize(filename)
        self.probeInfo = self._probe()
        self.rotationAngle = '0'
        viddict = {}
        if 'video' in self.probeInfo:
            viddict = self.probeInfo['video']
        self.inputfps = -1
        if '-r' in inputdict:
            self.inputfps = np.int(inputdict['-r'])
        elif self.INFO_AVERAGE_FRAMERATE in viddict:
            frtxt = viddict[self.INFO_AVERAGE_FRAMERATE]
            parts = frtxt.split('/')
            if len(parts) > 1:
                if np.float(parts[1]) == 0.0:
                    self.inputfps = self.DEFAULT_FRAMERATE
                else:
                    self.inputfps = np.float(parts[0]) / np.float(parts[1])
            else:
                self.inputfps = np.float(frtxt)
        else:
            self.inputfps = self.DEFAULT_FRAMERATE
        if 'tag' in viddict:
            tagdata = viddict['tag']
            if not isinstance(tagdata, list):
                tagdata = [tagdata]
            for tags in tagdata:
                if tags['@key'] == 'rotate':
                    self.rotationAngle = tags['@value']
        if '-s' in inputdict:
            widthheight = inputdict['-s'].split('x')
            self.inputwidth = np.int(widthheight[0])
            self.inputheight = np.int(widthheight[1])
        elif self.INFO_WIDTH in viddict and self.INFO_HEIGHT in viddict:
            self.inputwidth = np.int(viddict[self.INFO_WIDTH])
            self.inputheight = np.int(viddict[self.INFO_HEIGHT])
        else:
            raise ValueError('No way to determine width or height from video. Need `-s` in `inputdict`. Consult documentation on I/O.')
        if self.rotationAngle == '90' or self.rotationAngle == '270':
            self.inputwidth, self.inputheight = (self.inputheight, self.inputwidth)
        self.bpp = -1
        self.pix_fmt = ''
        if '-pix_fmt' in inputdict:
            self.pix_fmt = inputdict['-pix_fmt']
        elif self.INFO_PIX_FMT in viddict:
            self.pix_fmt = viddict[self.INFO_PIX_FMT]
        else:
            self.pix_fmt = self.DEFAULT_INPUT_PIX_FMT
            if verbosity != 0:
                warnings.warn('No input color space detected. Assuming {}.'.format(self.DEFAULT_INPUT_PIX_FMT), UserWarning)
        self.inputdepth = np.int(bpplut[self.pix_fmt][0])
        self.bpp = np.int(bpplut[self.pix_fmt][1])
        israw = str.encode(self.extension) in [b'.raw', b'.yuv']
        iswebcam = not os.path.isfile(filename)
        if '-vframes' in outputdict:
            self.inputframenum = np.int(outputdict['-vframes'])
        elif '-r' in outputdict:
            inputfps = np.int(outputdict['-r'])
            inputduration = np.float(viddict[self.INFO_DURATION])
            self.inputframenum = np.int(round(inputfps * inputduration) + 1)
        elif self.INFO_NB_FRAMES in viddict:
            self.inputframenum = np.int(viddict[self.INFO_NB_FRAMES])
        elif israw:
            self.inputframenum = np.int(self.size / (self.inputwidth * self.inputheight * (self.bpp / 8.0)))
        elif iswebcam:
            self.inputframenum = 0
        else:
            self.inputframenum = self._probCountFrames()
            if verbosity != 0:
                warnings.warn('Cannot determine frame count. Scanning input file, this is slow when repeated many times. Need `-vframes` in inputdict. Consult documentation on I/O.', UserWarning)
        if israw or iswebcam:
            inputdict['-pix_fmt'] = self.pix_fmt
        else:
            decoders = self._getSupportedDecoders()
            if decoders != NotImplemented:
                assert str.encode(self.extension).lower() in decoders, 'Unknown decoder extension: ' + self.extension.lower()
        if '-f' not in outputdict:
            outputdict['-f'] = self.OUTPUT_METHOD
        if '-pix_fmt' not in outputdict:
            outputdict['-pix_fmt'] = 'rgb24'
        self.output_pix_fmt = outputdict['-pix_fmt']
        if '-s' in outputdict:
            widthheight = outputdict['-s'].split('x')
            self.outputwidth = np.int(widthheight[0])
            self.outputheight = np.int(widthheight[1])
        else:
            self.outputwidth = self.inputwidth
            self.outputheight = self.inputheight
        self.outputdepth = np.int(bpplut[outputdict['-pix_fmt']][0])
        self.outputbpp = np.int(bpplut[outputdict['-pix_fmt']][1])
        bitpercomponent = self.outputbpp // self.outputdepth
        if bitpercomponent == 8:
            self.dtype = np.dtype('u1')
        elif bitpercomponent == 16:
            suffix = outputdict['-pix_fmt'][-2:]
            if suffix == 'le':
                self.dtype = np.dtype('<u2')
            elif suffix == 'be':
                self.dtype = np.dtype('>u2')
        else:
            raise ValueError(outputdict['-pix_fmt'] + 'is not a valid pix_fmt for numpy conversion')
        self._createProcess(inputdict, outputdict, verbosity)

    def __next__(self):
        return next(self.nextFrame())

    def __iter__(self):
        for frame in self.nextFrame():
            yield frame

    def _createProcess(self, inputdict, outputdict, verbosity):
        pass

    def _probCountFrames(self):
        return NotImplemented

    def _probe(self):
        pass

    def _getSupportedDecoders(self):
        return NotImplemented

    def _dict2Args(self, dict):
        args = []
        for key in dict.keys():
            args.append(key)
            args.append(dict[key])
        return args

    def getShape(self):
        """Returns a tuple (T, M, N, C)

        Returns the video shape in number of frames, height, width, and channels per pixel.
        """
        return (self.inputframenum, self.outputheight, self.outputwidth, self.outputdepth)

    def close(self):
        if self._proc is not None and self._proc.poll() is None:
            self._proc.stdin.close()
            self._proc.stdout.close()
            self._proc.stderr.close()
            self._terminate(0.2)
        self._proc = None

    def _terminate(self, timeout=1.0):
        """ Terminate the sub process.
        """
        if self._proc is None:
            return
        if self._proc.poll() is not None:
            return
        self._proc.terminate()
        etime = time.time() + timeout
        while time.time() < etime:
            time.sleep(0.01)
            if self._proc.poll() is not None:
                break

    def _read_frame_data(self):
        framesize = self.outputdepth * self.outputwidth * self.outputheight
        assert self._proc is not None
        try:
            arr = np.frombuffer(self._proc.stdout.read(framesize * self.dtype.itemsize), dtype=self.dtype)
            if len(arr) != framesize:
                return np.array([])
        except Exception as err:
            self._terminate()
            err1 = str(err)
            raise RuntimeError('%s' % (err1,))
        return arr

    def _readFrame(self):
        frame = self._read_frame_data()
        if len(frame) == 0:
            return frame
        if self.output_pix_fmt == 'rgb24':
            self._lastread = frame.reshape((self.outputheight, self.outputwidth, self.outputdepth))
        elif self.output_pix_fmt.startswith('yuv444p') or self.output_pix_fmt.startswith('yuvj444p') or self.output_pix_fmt.startswith('yuva444p'):
            self._lastread = frame.reshape((self.outputdepth, self.outputheight, self.outputwidth)).transpose((1, 2, 0))
        else:
            if self.verbosity > 0:
                warnings.warn('Unsupported reshaping from raw buffer to images frames  for format {:}. Assuming HEIGHTxWIDTHxCOLOR'.format(self.output_pix_fmt), UserWarning)
            self._lastread = frame.reshape((self.outputheight, self.outputwidth, self.outputdepth))
        return self._lastread

    def nextFrame(self):
        """Yields frames using a generator

        Returns T ndarrays of size (M, N, C), where T is number of frames,
        M is height, N is width, and C is number of channels per pixel.

        """
        if self.inputframenum == 0:
            while True:
                frame = self._readFrame()
                if len(frame) == 0:
                    break
                yield frame
        else:
            for i in range(self.inputframenum):
                frame = self._readFrame()
                if len(frame) == 0:
                    break
                yield frame

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()