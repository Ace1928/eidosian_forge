from kivy.clock import Clock
from kivy.logger import Logger
from kivy.core.audio import Sound, SoundLoader
from kivy.weakmethod import WeakMethod
import time

FFmpeg based audio player
=========================

To use, you need to install ffpyplayer and have a compiled ffmpeg shared
library.

    https://github.com/matham/ffpyplayer

The docs there describe how to set this up. But briefly, first you need to
compile ffmpeg using the shared flags while disabling the static flags (you'll
probably have to set the fPIC flag, e.g. CFLAGS=-fPIC). Here's some
instructions: https://trac.ffmpeg.org/wiki/CompilationGuide. For Windows, you
can download compiled GPL binaries from http://ffmpeg.zeranoe.com/builds/.
Similarly, you should download SDL.

Now, you should a ffmpeg and sdl directory. In each, you should have a include,
bin, and lib directory, where e.g. for Windows, lib contains the .dll.a files,
while bin contains the actual dlls. The include directory holds the headers.
The bin directory is only needed if the shared libraries are not already on
the path. In the environment define FFMPEG_ROOT and SDL_ROOT, each pointing to
the ffmpeg, and SDL directories, respectively. (If you're using SDL2,
the include directory will contain a directory called SDL2, which then holds
the headers).

Once defined, download the ffpyplayer git and run

    python setup.py build_ext --inplace

Finally, before running you need to ensure that ffpyplayer is in python's path.

..Note::

    When kivy exits by closing the window while the audio is playing,
    it appears that the __del__method of SoundFFPy
    is not called. Because of this the SoundFFPy object is not
    properly deleted when kivy exits. The consequence is that because
    MediaPlayer creates internal threads which do not have their daemon
    flag set, when the main threads exists it'll hang and wait for the other
    MediaPlayer threads to exit. But since __del__ is not called to delete the
    MediaPlayer object, those threads will remain alive hanging kivy. What this
    means is that you have to be sure to delete the MediaPlayer object before
    kivy exits by setting it to None.
