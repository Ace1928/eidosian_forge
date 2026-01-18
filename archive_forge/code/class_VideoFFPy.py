from threading import Thread
from queue import Queue, Empty, Full
from kivy.clock import Clock, mainthread
from kivy.logger import Logger
from kivy.core.video import VideoBase
from kivy.graphics import Rectangle, BindTexture
from kivy.graphics.texture import Texture
from kivy.graphics.fbo import Fbo
from kivy.weakmethod import WeakMethod
import time
class VideoFFPy(VideoBase):
    YUV_RGB_FS = '\n    $HEADER$\n    uniform sampler2D tex_y;\n    uniform sampler2D tex_u;\n    uniform sampler2D tex_v;\n\n    void main(void) {\n        float y = texture2D(tex_y, tex_coord0).r;\n        float u = texture2D(tex_u, tex_coord0).r - 0.5;\n        float v = texture2D(tex_v, tex_coord0).r - 0.5;\n        float r = y +             1.402 * v;\n        float g = y - 0.344 * u - 0.714 * v;\n        float b = y + 1.772 * u;\n        gl_FragColor = vec4(r, g, b, 1.0);\n    }\n    '
    _trigger = None

    def __init__(self, **kwargs):
        self._ffplayer = None
        self._thread = None
        self._next_frame = None
        self._seek_queue = []
        self._ffplayer_need_quit = False
        self._wakeup_queue = Queue(maxsize=1)
        self._trigger = Clock.create_trigger(self._redraw)
        super(VideoFFPy, self).__init__(**kwargs)

    @property
    def _is_stream(self):
        return self.filename.startswith('rtsp://')

    def __del__(self):
        self.unload()

    def _wakeup_thread(self):
        try:
            self._wakeup_queue.put(None, False)
        except Full:
            pass

    def _wait_for_wakeup(self, timeout):
        try:
            self._wakeup_queue.get(True, timeout)
        except Empty:
            pass

    def _player_callback(self, selector, value):
        if self._ffplayer is None:
            return
        if selector == 'quit':

            def close(*args):
                self.unload()
            Clock.schedule_once(close, 0)

    def _get_position(self):
        if self._ffplayer is not None:
            return self._ffplayer.get_pts()
        return 0

    def _set_position(self, pos):
        self.seek(pos)

    def _set_volume(self, volume):
        self._volume = volume
        if self._ffplayer is not None:
            self._ffplayer.set_volume(self._volume)

    def _get_duration(self):
        if self._ffplayer is None:
            return 0
        return self._ffplayer.get_metadata()['duration']

    @mainthread
    def _do_eos(self):
        if self.eos == 'pause':
            self.pause()
        elif self.eos == 'stop':
            self.stop()
        elif self.eos == 'loop':
            self.position = 0
        self.dispatch('on_eos')

    @mainthread
    def _finish_setup(self):
        if self._ffplayer is not None:
            self._ffplayer.set_volume(self._volume)
            self._ffplayer.set_pause(self._state == 'paused')
            self._wakeup_thread()

    def _redraw(self, *args):
        if not self._ffplayer:
            return
        next_frame = self._next_frame
        if not next_frame:
            return
        img, pts = next_frame
        if img.get_size() != self._size or self._texture is None:
            self._size = w, h = img.get_size()
            if self._out_fmt == 'yuv420p':
                w2 = int(w / 2)
                h2 = int(h / 2)
                self._tex_y = Texture.create(size=(w, h), colorfmt='luminance')
                self._tex_u = Texture.create(size=(w2, h2), colorfmt='luminance')
                self._tex_v = Texture.create(size=(w2, h2), colorfmt='luminance')
                self._fbo = fbo = Fbo(size=self._size)
                with fbo:
                    BindTexture(texture=self._tex_u, index=1)
                    BindTexture(texture=self._tex_v, index=2)
                    Rectangle(size=fbo.size, texture=self._tex_y)
                fbo.shader.fs = VideoFFPy.YUV_RGB_FS
                fbo['tex_y'] = 0
                fbo['tex_u'] = 1
                fbo['tex_v'] = 2
                self._texture = fbo.texture
            else:
                self._texture = Texture.create(size=self._size, colorfmt='rgba')
            self._texture.flip_vertical()
            self.dispatch('on_load')
        if self._texture:
            if self._out_fmt == 'yuv420p':
                dy, du, dv, _ = img.to_memoryview()
                if dy and du and dv:
                    self._tex_y.blit_buffer(dy, colorfmt='luminance')
                    self._tex_u.blit_buffer(du, colorfmt='luminance')
                    self._tex_v.blit_buffer(dv, colorfmt='luminance')
                    self._fbo.ask_update()
                    self._fbo.draw()
            else:
                self._texture.blit_buffer(img.to_memoryview()[0], colorfmt='rgba')
            self.dispatch('on_frame')

    def _next_frame_run(self, ffplayer):
        sleep = time.sleep
        trigger = self._trigger
        did_dispatch_eof = False
        wait_for_wakeup = self._wait_for_wakeup
        seek_queue = self._seek_queue
        while not self._ffplayer_need_quit:
            src_pix_fmt = ffplayer.get_metadata().get('src_pix_fmt')
            if not src_pix_fmt:
                wait_for_wakeup(0.005)
                continue
            if src_pix_fmt in (b'yuv420p', 'yuv420p'):
                self._out_fmt = 'yuv420p'
                ffplayer.set_output_pix_fmt(self._out_fmt)
            break
        if self._ffplayer_need_quit:
            ffplayer.close_player()
            return
        self._ffplayer = ffplayer
        self._finish_setup()
        while not self._ffplayer_need_quit:
            seek_happened = False
            if seek_queue:
                vals = seek_queue[:]
                del seek_queue[:len(vals)]
                percent, precise = vals[-1]
                ffplayer.seek(percent * ffplayer.get_metadata()['duration'], relative=False, accurate=precise)
                seek_happened = True
                did_dispatch_eof = False
                self._next_frame = None
            if seek_happened and ffplayer.get_pause():
                ffplayer.set_volume(0.0)
                ffplayer.set_pause(False)
                try:
                    to_skip = 6
                    while True:
                        frame, val = ffplayer.get_frame(show=False)
                        if val in ('paused', 'eof'):
                            break
                        if seek_queue:
                            break
                        if frame is None:
                            sleep(0.005)
                            continue
                        to_skip -= 1
                        if to_skip == 0:
                            break
                    frame, val = ffplayer.get_frame(force_refresh=True)
                finally:
                    ffplayer.set_pause(bool(self._state == 'paused'))
                    ffplayer.set_volume(self._volume)
            else:
                frame, val = ffplayer.get_frame()
            if val == 'eof':
                if not did_dispatch_eof:
                    self._do_eos()
                    did_dispatch_eof = True
                wait_for_wakeup(None)
            elif val == 'paused':
                did_dispatch_eof = False
                wait_for_wakeup(None)
            else:
                did_dispatch_eof = False
                if frame:
                    self._next_frame = frame
                    trigger()
                else:
                    val = val if val else 1 / 30.0
                wait_for_wakeup(val)
        ffplayer.close_player()

    def seek(self, percent, precise=True):
        self._seek_queue.append((percent, precise))
        self._wakeup_thread()

    def stop(self):
        self.unload()

    def pause(self):
        if self._state == 'playing':
            if self._ffplayer is not None:
                self._ffplayer.set_pause(True)
            self._state = 'paused'
            self._wakeup_thread()

    def play(self):
        if self._ffplayer:
            assert self._state in ('paused', 'playing')
            if self._state == 'paused':
                self._ffplayer.set_pause(False)
                self._state = 'playing'
                self._wakeup_thread()
            return
        if self._state == 'playing':
            return
        elif self._state == 'paused':
            self._state = 'playing'
            self._wakeup_thread()
            return
        self.load()
        self._out_fmt = 'rgba'
        ff_opts = {'paused': not self._is_stream, 'out_fmt': self._out_fmt, 'sn': True, 'volume': self._volume}
        ffplayer = MediaPlayer(self._filename, callback=self._player_callback, thread_lib='SDL', loglevel='info', ff_opts=ff_opts)
        self._thread = Thread(target=self._next_frame_run, name='Next frame', args=(ffplayer,))
        self._thread.daemon = True
        self._state = 'playing'
        self._thread.start()

    def load(self):
        self.unload()

    def unload(self):
        self._wakeup_thread()
        self._ffplayer_need_quit = True
        if self._thread:
            self._thread.join()
            self._thread = None
        if self._ffplayer:
            self._ffplayer = None
        self._next_frame = None
        self._size = (0, 0)
        self._state = ''
        self._seek_queue = []
        self._ffplayer_need_quit = False
        self._wakeup_queue = Queue(maxsize=1)