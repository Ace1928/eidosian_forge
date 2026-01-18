import os
import subprocess as sp
import tempfile
import warnings
import numpy as np
import proglog
from imageio import imread, imsave
from ..Clip import Clip
from ..compat import DEVNULL, string_types
from ..config import get_setting
from ..decorators import (add_mask_if_none, apply_to_mask,
from ..tools import (deprecated_version_of, extensions_dict, find_extension,
from .io.ffmpeg_writer import ffmpeg_write_video
from .io.gif_writers import (write_gif, write_gif_with_image_io,
from .tools.drawing import blit
class VideoClip(Clip):
    """Base class for video clips.

    See ``VideoFileClip``, ``ImageClip`` etc. for more user-friendly
    classes.


    Parameters
    -----------

    ismask
      `True` if the clip is going to be used as a mask.


    Attributes
    ----------

    size
      The size of the clip, (width,heigth), in pixels.

    w, h
      The width and height of the clip, in pixels.

    ismask
      Boolean set to `True` if the clip is a mask.

    make_frame
      A function ``t-> frame at time t`` where ``frame`` is a
      w*h*3 RGB array.

    mask (default None)
      VideoClip mask attached to this clip. If mask is ``None``,
                The video clip is fully opaque.

    audio (default None)
      An AudioClip instance containing the audio of the video clip.

    pos
      A function ``t->(x,y)`` where ``x,y`` is the position
      of the clip when it is composed with other clips.
      See ``VideoClip.set_pos`` for more details

    relative_pos
      See variable ``pos``.

    """

    def __init__(self, make_frame=None, ismask=False, duration=None, has_constant_size=True):
        Clip.__init__(self)
        self.mask = None
        self.audio = None
        self.pos = lambda t: (0, 0)
        self.relative_pos = False
        if make_frame:
            self.make_frame = make_frame
            self.size = self.get_frame(0).shape[:2][::-1]
        self.ismask = ismask
        self.has_constant_size = has_constant_size
        if duration is not None:
            self.duration = duration
            self.end = duration

    @property
    def w(self):
        return self.size[0]

    @property
    def h(self):
        return self.size[1]

    @property
    def aspect_ratio(self):
        return self.w / float(self.h)

    @convert_to_seconds(['t'])
    @convert_masks_to_RGB
    def save_frame(self, filename, t=0, withmask=True):
        """ Save a clip's frame to an image file.

        Saves the frame of clip corresponding to time ``t`` in
        'filename'. ``t`` can be expressed in seconds (15.35), in
        (min, sec), in (hour, min, sec), or as a string: '01:03:05.35'.

        If ``withmask`` is ``True`` the mask is saved in
        the alpha layer of the picture (only works with PNGs).

        """
        im = self.get_frame(t)
        if withmask and self.mask is not None:
            mask = 255 * self.mask.get_frame(t)
            im = np.dstack([im, mask]).astype('uint8')
        else:
            im = im.astype('uint8')
        imsave(filename, im)

    @requires_duration
    @use_clip_fps_by_default
    @convert_masks_to_RGB
    def write_videofile(self, filename, fps=None, codec=None, bitrate=None, audio=True, audio_fps=44100, preset='medium', audio_nbytes=4, audio_codec=None, audio_bitrate=None, audio_bufsize=2000, temp_audiofile=None, rewrite_audio=True, remove_temp=True, write_logfile=False, verbose=True, threads=None, ffmpeg_params=None, logger='bar'):
        """Write the clip to a videofile.

        Parameters
        -----------

        filename
          Name of the video file to write in.
          The extension must correspond to the "codec" used (see below),
          or simply be '.avi' (which will work with any codec).

        fps
          Number of frames per second in the resulting video file. If None is
          provided, and the clip has an fps attribute, this fps will be used.

        codec
          Codec to use for image encoding. Can be any codec supported
          by ffmpeg. If the filename is has extension '.mp4', '.ogv', '.webm',
          the codec will be set accordingly, but you can still set it if you
          don't like the default. For other extensions, the output filename
          must be set accordingly.

          Some examples of codecs are:

          ``'libx264'`` (default codec for file extension ``.mp4``)
          makes well-compressed videos (quality tunable using 'bitrate').


          ``'mpeg4'`` (other codec for extension ``.mp4``) can be an alternative
          to ``'libx264'``, and produces higher quality videos by default.


          ``'rawvideo'`` (use file extension ``.avi``) will produce
          a video of perfect quality, of possibly very huge size.


          ``png`` (use file extension ``.avi``) will produce a video
          of perfect quality, of smaller size than with ``rawvideo``.


          ``'libvorbis'`` (use file extension ``.ogv``) is a nice video
          format, which is completely free/ open source. However not
          everyone has the codecs installed by default on their machine.


          ``'libvpx'`` (use file extension ``.webm``) is tiny a video
          format well indicated for web videos (with HTML5). Open source.


        audio
          Either ``True``, ``False``, or a file name.
          If ``True`` and the clip has an audio clip attached, this
          audio clip will be incorporated as a soundtrack in the movie.
          If ``audio`` is the name of an audio file, this audio file
          will be incorporated as a soundtrack in the movie.

        audiofps
          frame rate to use when generating the sound.

        temp_audiofile
          the name of the temporary audiofile to be generated and
          incorporated in the the movie, if any.

        audio_codec
          Which audio codec should be used. Examples are 'libmp3lame'
          for '.mp3', 'libvorbis' for 'ogg', 'libfdk_aac':'m4a',
          'pcm_s16le' for 16-bit wav and 'pcm_s32le' for 32-bit wav.
          Default is 'libmp3lame', unless the video extension is 'ogv'
          or 'webm', at which case the default is 'libvorbis'.

        audio_bitrate
          Audio bitrate, given as a string like '50k', '500k', '3000k'.
          Will determine the size/quality of audio in the output file.
          Note that it mainly an indicative goal, the bitrate won't
          necessarily be the this in the final file.

        preset
          Sets the time that FFMPEG will spend optimizing the compression.
          Choices are: ultrafast, superfast, veryfast, faster, fast, medium,
          slow, slower, veryslow, placebo. Note that this does not impact
          the quality of the video, only the size of the video file. So
          choose ultrafast when you are in a hurry and file size does not
          matter.

        threads
          Number of threads to use for ffmpeg. Can speed up the writing of
          the video on multicore computers.

        ffmpeg_params
          Any additional ffmpeg parameters you would like to pass, as a list
          of terms, like ['-option1', 'value1', '-option2', 'value2'].

        write_logfile
          If true, will write log files for the audio and the video.
          These will be files ending with '.log' with the name of the
          output file in them.

        logger
          Either "bar" for progress bar or None or any Proglog logger.

        verbose (deprecated, kept for compatibility)
          Formerly used for toggling messages on/off. Use logger=None now.

        Examples
        ========

        >>> from moviepy.editor import VideoFileClip
        >>> clip = VideoFileClip("myvideo.mp4").subclip(100,120)
        >>> clip.write_videofile("my_new_video.mp4")
        >>> clip.close()

        """
        name, ext = os.path.splitext(os.path.basename(filename))
        ext = ext[1:].lower()
        logger = proglog.default_bar_logger(logger)
        if codec is None:
            try:
                codec = extensions_dict[ext]['codec'][0]
            except KeyError:
                raise ValueError("MoviePy couldn't find the codec associated with the filename. Provide the 'codec' parameter in write_videofile.")
        if audio_codec is None:
            if ext in ['ogv', 'webm']:
                audio_codec = 'libvorbis'
            else:
                audio_codec = 'libmp3lame'
        elif audio_codec == 'raw16':
            audio_codec = 'pcm_s16le'
        elif audio_codec == 'raw32':
            audio_codec = 'pcm_s32le'
        audiofile = audio if is_string(audio) else None
        make_audio = audiofile is None and audio == True and (self.audio is not None)
        if make_audio and temp_audiofile:
            audiofile = temp_audiofile
        elif make_audio:
            audio_ext = find_extension(audio_codec)
            audiofile = name + Clip._TEMP_FILES_PREFIX + 'wvf_snd.%s' % audio_ext
        logger(message='Moviepy - Building video %s.' % filename)
        if make_audio:
            self.audio.write_audiofile(audiofile, audio_fps, audio_nbytes, audio_bufsize, audio_codec, bitrate=audio_bitrate, write_logfile=write_logfile, verbose=verbose, logger=logger)
        ffmpeg_write_video(self, filename, fps, codec, bitrate=bitrate, preset=preset, write_logfile=write_logfile, audiofile=audiofile, verbose=verbose, threads=threads, ffmpeg_params=ffmpeg_params, logger=logger)
        if remove_temp and make_audio:
            if os.path.exists(audiofile):
                os.remove(audiofile)
        logger(message='Moviepy - video ready %s' % filename)

    @requires_duration
    @use_clip_fps_by_default
    @convert_masks_to_RGB
    def write_images_sequence(self, nameformat, fps=None, verbose=True, withmask=True, logger='bar'):
        """ Writes the videoclip to a sequence of image files.

        Parameters
        -----------

        nameformat
          A filename specifying the numerotation format and extension
          of the pictures. For instance "frame%03d.png" for filenames
          indexed with 3 digits and PNG format. Also possible:
          "some_folder/frame%04d.jpeg", etc.

        fps
          Number of frames per second to consider when writing the
          clip. If not specified, the clip's ``fps`` attribute will
          be used if it has one.

        withmask
          will save the clip's mask (if any) as an alpha canal (PNGs only).

        verbose
          Boolean indicating whether to print information.

        logger
          Either 'bar' (progress bar) or None or any Proglog logger.


        Returns
        --------

        names_list
          A list of all the files generated.

        Notes
        ------

        The resulting image sequence can be read using e.g. the class
        ``ImageSequenceClip``.

        """
        logger = proglog.default_bar_logger(logger)
        logger(message='Moviepy - Writing frames %s.' % nameformat)
        tt = np.arange(0, self.duration, 1.0 / fps)
        filenames = []
        for i, t in logger.iter_bar(t=list(enumerate(tt))):
            name = nameformat % i
            filenames.append(name)
            self.save_frame(name, t, withmask=withmask)
        logger(message='Moviepy - Done writing frames %s.' % nameformat)
        return filenames

    @requires_duration
    @convert_masks_to_RGB
    def write_gif(self, filename, fps=None, program='imageio', opt='nq', fuzz=1, verbose=True, loop=0, dispose=False, colors=None, tempfiles=False, logger='bar'):
        """ Write the VideoClip to a GIF file.

        Converts a VideoClip into an animated GIF using ImageMagick
        or ffmpeg.

        Parameters
        -----------

        filename
          Name of the resulting gif file.

        fps
          Number of frames per second (see note below). If it
          isn't provided, then the function will look for the clip's
          ``fps`` attribute (VideoFileClip, for instance, have one).

        program
          Software to use for the conversion, either 'imageio' (this will use
          the library FreeImage through ImageIO), or 'ImageMagick', or 'ffmpeg'.

        opt
          Optimalization to apply. If program='imageio', opt must be either 'wu'
          (Wu) or 'nq' (Neuquant). If program='ImageMagick',
          either 'optimizeplus' or 'OptimizeTransparency'.

        fuzz
          (ImageMagick only) Compresses the GIF by considering that
          the colors that are less than fuzz% different are in fact
          the same.

        tempfiles
          Writes every frame to a file instead of passing them in the RAM.
          Useful on computers with little RAM. Can only be used with
          ImageMagick' or 'ffmpeg'.

        progress_bar
          If True, displays a progress bar


        Notes
        -----

        The gif will be playing the clip in real time (you can
        only change the frame rate). If you want the gif to be played
        slower than the clip you will use ::

            >>> # slow down clip 50% and make it a gif
            >>> myClip.speedx(0.5).to_gif('myClip.gif')

        """
        if program == 'imageio':
            write_gif_with_image_io(self, filename, fps=fps, opt=opt, loop=loop, verbose=verbose, colors=colors, logger=logger)
        elif tempfiles:
            opt = 'optimizeplus' if opt == 'nq' else 'OptimizeTransparency'
            write_gif_with_tempfiles(self, filename, fps=fps, program=program, opt=opt, fuzz=fuzz, verbose=verbose, loop=loop, dispose=dispose, colors=colors, logger=logger)
        else:
            opt = 'optimizeplus' if opt == 'nq' else 'OptimizeTransparency'
            write_gif(self, filename, fps=fps, program=program, opt=opt, fuzz=fuzz, verbose=verbose, loop=loop, dispose=dispose, colors=colors, logger=logger)

    def subfx(self, fx, ta=0, tb=None, **kwargs):
        """Apply a transformation to a part of the clip.

        Returns a new clip in which the function ``fun`` (clip->clip)
        has been applied to the subclip between times `ta` and `tb`
        (in seconds).

        Examples
        ---------

        >>> # The scene between times t=3s and t=6s in ``clip`` will be
        >>> # be played twice slower in ``newclip``
        >>> newclip = clip.subapply(lambda c:c.speedx(0.5) , 3,6)

        """
        left = self.subclip(0, ta) if ta else None
        center = self.subclip(ta, tb).fx(fx, **kwargs)
        right = self.subclip(t_start=tb) if tb else None
        clips = [c for c in (left, center, right) if c]
        from moviepy.video.compositing.concatenate import concatenate_videoclips
        return concatenate_videoclips(clips).set_start(self.start)

    def fl_image(self, image_func, apply_to=None):
        """
        Modifies the images of a clip by replacing the frame
        `get_frame(t)` by another frame,  `image_func(get_frame(t))`
        """
        apply_to = apply_to or []
        return self.fl(lambda gf, t: image_func(gf(t)), apply_to)

    def fill_array(self, pre_array, shape=(0, 0)):
        pre_shape = pre_array.shape
        dx = shape[0] - pre_shape[0]
        dy = shape[1] - pre_shape[1]
        post_array = pre_array
        if dx < 0:
            post_array = pre_array[:shape[0]]
        elif dx > 0:
            x_1 = [[[1, 1, 1]] * pre_shape[1]] * dx
            post_array = np.vstack((pre_array, x_1))
        if dy < 0:
            post_array = post_array[:, :shape[1]]
        elif dy > 0:
            x_1 = [[[1, 1, 1]] * dy] * post_array.shape[0]
            post_array = np.hstack((post_array, x_1))
        return post_array

    def blit_on(self, picture, t):
        """
        Returns the result of the blit of the clip's frame at time `t`
        on the given `picture`, the position of the clip being given
        by the clip's ``pos`` attribute. Meant for compositing.
        """
        hf, wf = framesize = picture.shape[:2]
        if self.ismask and picture.max():
            return np.minimum(1, picture + self.blit_on(np.zeros(framesize), t))
        ct = t - self.start
        img = self.get_frame(ct)
        mask = self.mask.get_frame(ct) if self.mask else None
        if mask is not None and (img.shape[0] != mask.shape[0] or img.shape[1] != mask.shape[1]):
            img = self.fill_array(img, mask.shape)
        hi, wi = img.shape[:2]
        pos = self.pos(ct)
        if isinstance(pos, str):
            pos = {'center': ['center', 'center'], 'left': ['left', 'center'], 'right': ['right', 'center'], 'top': ['center', 'top'], 'bottom': ['center', 'bottom']}[pos]
        else:
            pos = list(pos)
        if self.relative_pos:
            for i, dim in enumerate([wf, hf]):
                if not isinstance(pos[i], str):
                    pos[i] = dim * pos[i]
        if isinstance(pos[0], str):
            D = {'left': 0, 'center': (wf - wi) / 2, 'right': wf - wi}
            pos[0] = D[pos[0]]
        if isinstance(pos[1], str):
            D = {'top': 0, 'center': (hf - hi) / 2, 'bottom': hf - hi}
            pos[1] = D[pos[1]]
        pos = map(int, pos)
        return blit(img, picture, pos, mask=mask, ismask=self.ismask)

    def add_mask(self):
        """Add a mask VideoClip to the VideoClip.

        Returns a copy of the clip with a completely opaque mask
        (made of ones). This makes computations slower compared to
        having a None mask but can be useful in many cases. Choose

        Set ``constant_size`` to  `False` for clips with moving
        image size.
        """
        if self.has_constant_size:
            mask = ColorClip(self.size, 1.0, ismask=True)
            return self.set_mask(mask.set_duration(self.duration))
        else:
            make_frame = lambda t: np.ones(self.get_frame(t).shape[:2], dtype=float)
            mask = VideoClip(ismask=True, make_frame=make_frame)
            return self.set_mask(mask.set_duration(self.duration))

    def on_color(self, size=None, color=(0, 0, 0), pos=None, col_opacity=None):
        """Place the clip on a colored background.

        Returns a clip made of the current clip overlaid on a color
        clip of a possibly bigger size. Can serve to flatten transparent
        clips.

        Parameters
        -----------

        size
          Size (width, height) in pixels of the final clip.
          By default it will be the size of the current clip.

        color
          Background color of the final clip ([R,G,B]).

        pos
          Position of the clip in the final clip. 'center' is the default

        col_opacity
          Parameter in 0..1 indicating the opacity of the colored
          background.

        """
        from .compositing.CompositeVideoClip import CompositeVideoClip
        if size is None:
            size = self.size
        if pos is None:
            pos = 'center'
        colorclip = ColorClip(size, color=color)
        if col_opacity is not None:
            colorclip = ColorClip(size, color=color, duration=self.duration).set_opacity(col_opacity)
            result = CompositeVideoClip([colorclip, self.set_position(pos)])
        else:
            result = CompositeVideoClip([self.set_position(pos)], size=size, bg_color=color)
        if isinstance(self, ImageClip) and (not hasattr(pos, '__call__')) and (self.mask is None or isinstance(self.mask, ImageClip)):
            new_result = result.to_ImageClip()
            if result.mask is not None:
                new_result.mask = result.mask.to_ImageClip()
            return new_result.set_duration(result.duration)
        return result

    @outplace
    def set_make_frame(self, mf):
        """Change the clip's ``get_frame``.

        Returns a copy of the VideoClip instance, with the make_frame
        attribute set to `mf`.
        """
        self.make_frame = mf
        self.size = self.get_frame(0).shape[:2][::-1]

    @outplace
    def set_audio(self, audioclip):
        """Attach an AudioClip to the VideoClip.

        Returns a copy of the VideoClip instance, with the `audio`
        attribute set to ``audio``, which must be an AudioClip instance.
        """
        self.audio = audioclip

    @outplace
    def set_mask(self, mask):
        """Set the clip's mask.

        Returns a copy of the VideoClip with the mask attribute set to
        ``mask``, which must be a greyscale (values in 0-1) VideoClip"""
        assert mask is None or mask.ismask
        self.mask = mask

    @add_mask_if_none
    @outplace
    def set_opacity(self, op):
        """Set the opacity/transparency level of the clip.

        Returns a semi-transparent copy of the clip where the mask is
        multiplied by ``op`` (any float, normally between 0 and 1).
        """
        self.mask = self.mask.fl_image(lambda pic: op * pic)

    @apply_to_mask
    @outplace
    def set_position(self, pos, relative=False):
        """Set the clip's position in compositions.

        Sets the position that the clip will have when included
        in compositions. The argument ``pos`` can be either a couple
        ``(x,y)`` or a function ``t-> (x,y)``. `x` and `y` mark the
        location of the top left corner of the clip, and can be
        of several types.

        Examples
        ----------

        >>> clip.set_position((45,150)) # x=45, y=150
        >>>
        >>> # clip horizontally centered, at the top of the picture
        >>> clip.set_position(("center","top"))
        >>>
        >>> # clip is at 40% of the width, 70% of the height:
        >>> clip.set_position((0.4,0.7), relative=True)
        >>>
        >>> # clip's position is horizontally centered, and moving up !
        >>> clip.set_position(lambda t: ('center', 50+t) )

        """
        self.relative_pos = relative
        if hasattr(pos, '__call__'):
            self.pos = pos
        else:
            self.pos = lambda t: pos

    @convert_to_seconds(['t'])
    def to_ImageClip(self, t=0, with_mask=True, duration=None):
        """
        Returns an ImageClip made out of the clip's frame at time ``t``,
        which can be expressed in seconds (15.35), in (min, sec),
        in (hour, min, sec), or as a string: '01:03:05.35'.
        """
        newclip = ImageClip(self.get_frame(t), ismask=self.ismask, duration=duration)
        if with_mask and self.mask is not None:
            newclip.mask = self.mask.to_ImageClip(t)
        return newclip

    def to_mask(self, canal=0):
        """Return a mask a video clip made from the clip."""
        if self.ismask:
            return self
        else:
            newclip = self.fl_image(lambda pic: 1.0 * pic[:, :, canal] / 255)
            newclip.ismask = True
            return newclip

    def to_RGB(self):
        """Return a non-mask video clip made from the mask video clip."""
        if self.ismask:
            f = lambda pic: np.dstack(3 * [255 * pic]).astype('uint8')
            newclip = self.fl_image(f)
            newclip.ismask = False
            return newclip
        else:
            return self

    @outplace
    def without_audio(self):
        """Remove the clip's audio.

        Return a copy of the clip with audio set to None.

        """
        self.audio = None

    @outplace
    def afx(self, fun, *a, **k):
        """Transform the clip's audio.

        Return a new clip whose audio has been transformed by ``fun``.

        """
        self.audio = self.audio.fx(fun, *a, **k)