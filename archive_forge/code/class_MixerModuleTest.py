import sys
import os
import unittest
import pathlib
import platform
import time
from pygame.tests.test_utils import example_path
import pygame
from pygame import mixer
class MixerModuleTest(unittest.TestCase):

    def tearDown(self):
        mixer.quit()
        mixer.pre_init(0, 0, 0, 0)

    def test_init__keyword_args(self):
        mixer.init(**CONFIG)
        mixer_conf = mixer.get_init()
        self.assertEqual(mixer_conf[0], CONFIG['frequency'])
        self.assertEqual(abs(mixer_conf[1]), abs(CONFIG['size']))
        self.assertGreaterEqual(mixer_conf[2], CONFIG['channels'])

    def test_pre_init__keyword_args(self):
        mixer.pre_init(**CONFIG)
        mixer.init()
        mixer_conf = mixer.get_init()
        self.assertEqual(mixer_conf[0], CONFIG['frequency'])
        self.assertEqual(abs(mixer_conf[1]), abs(CONFIG['size']))
        self.assertGreaterEqual(mixer_conf[2], CONFIG['channels'])

    def test_pre_init__zero_values(self):
        mixer.pre_init(22050, -8, 1)
        mixer.pre_init(0, 0, 0)
        mixer.init(allowedchanges=0)
        self.assertEqual(mixer.get_init()[0], 44100)
        self.assertEqual(mixer.get_init()[1], -16)
        self.assertGreaterEqual(mixer.get_init()[2], 2)

    def test_init__zero_values(self):
        mixer.pre_init(44100, 8, 1, allowedchanges=0)
        mixer.init(0, 0, 0)
        self.assertEqual(mixer.get_init(), (44100, 8, 1))

    def test_get_init__returns_None_if_mixer_not_initialized(self):
        self.assertIsNone(mixer.get_init())

    def test_get_num_channels__defaults_eight_after_init(self):
        mixer.init()
        self.assertEqual(mixer.get_num_channels(), 8)

    def test_set_num_channels(self):
        mixer.init()
        default_num_channels = mixer.get_num_channels()
        for i in range(1, default_num_channels + 1):
            mixer.set_num_channels(i)
            self.assertEqual(mixer.get_num_channels(), i)

    def test_quit(self):
        """get_num_channels() Should throw pygame.error if uninitialized
        after mixer.quit()"""
        mixer.init()
        mixer.quit()
        self.assertRaises(pygame.error, mixer.get_num_channels)

    @unittest.skipIf(sys.platform.startswith('win'), 'See github issue 892.')
    @unittest.skipIf(IS_PYPY, 'random errors here with pypy')
    def test_sound_args(self):

        def get_bytes(snd):
            return snd.get_raw()
        mixer.init()
        sample = b'\x00\xff' * 24
        wave_path = example_path(os.path.join('data', 'house_lo.wav'))
        uwave_path = str(wave_path)
        bwave_path = uwave_path.encode(sys.getfilesystemencoding())
        snd = mixer.Sound(file=wave_path)
        self.assertTrue(snd.get_length() > 0.5)
        snd_bytes = get_bytes(snd)
        self.assertTrue(len(snd_bytes) > 1000)
        self.assertEqual(get_bytes(mixer.Sound(wave_path)), snd_bytes)
        self.assertEqual(get_bytes(mixer.Sound(file=uwave_path)), snd_bytes)
        self.assertEqual(get_bytes(mixer.Sound(uwave_path)), snd_bytes)
        arg_emsg = 'Sound takes either 1 positional or 1 keyword argument'
        with self.assertRaises(TypeError) as cm:
            mixer.Sound()
        self.assertEqual(str(cm.exception), arg_emsg)
        with self.assertRaises(TypeError) as cm:
            mixer.Sound(wave_path, buffer=sample)
        self.assertEqual(str(cm.exception), arg_emsg)
        with self.assertRaises(TypeError) as cm:
            mixer.Sound(sample, file=wave_path)
        self.assertEqual(str(cm.exception), arg_emsg)
        with self.assertRaises(TypeError) as cm:
            mixer.Sound(buffer=sample, file=wave_path)
        self.assertEqual(str(cm.exception), arg_emsg)
        with self.assertRaises(TypeError) as cm:
            mixer.Sound(foobar=sample)
        self.assertEqual(str(cm.exception), "Unrecognized keyword argument 'foobar'")
        snd = mixer.Sound(wave_path, **{})
        self.assertEqual(get_bytes(snd), snd_bytes)
        snd = mixer.Sound(*[], **{'file': wave_path})
        with self.assertRaises(TypeError) as cm:
            mixer.Sound([])
        self.assertEqual(str(cm.exception), 'Unrecognized argument (type list)')
        with self.assertRaises(TypeError) as cm:
            snd = mixer.Sound(buffer=[])
        emsg = 'Expected object with buffer interface: got a list'
        self.assertEqual(str(cm.exception), emsg)
        ufake_path = '12345678'
        self.assertRaises(IOError, mixer.Sound, ufake_path)
        self.assertRaises(IOError, mixer.Sound, '12345678')
        with self.assertRaises(TypeError) as cm:
            mixer.Sound(buffer='something')
        emsg = 'Unicode object not allowed as buffer object'
        self.assertEqual(str(cm.exception), emsg)
        self.assertEqual(get_bytes(mixer.Sound(buffer=sample)), sample)
        if type(sample) != str:
            somebytes = get_bytes(mixer.Sound(sample))
            self.assertEqual(somebytes, sample)
        self.assertEqual(get_bytes(mixer.Sound(file=bwave_path)), snd_bytes)
        self.assertEqual(get_bytes(mixer.Sound(bwave_path)), snd_bytes)
        snd = mixer.Sound(wave_path)
        with self.assertRaises(TypeError) as cm:
            mixer.Sound(wave_path, array=snd)
        self.assertEqual(str(cm.exception), arg_emsg)
        with self.assertRaises(TypeError) as cm:
            mixer.Sound(buffer=sample, array=snd)
        self.assertEqual(str(cm.exception), arg_emsg)
        snd2 = mixer.Sound(array=snd)
        self.assertEqual(snd.get_raw(), snd2.get_raw())

    def test_sound_unicode(self):
        """test non-ASCII unicode path"""
        mixer.init()
        import shutil
        ep = example_path('data')
        temp_file = os.path.join(ep, '你好.wav')
        org_file = os.path.join(ep, 'house_lo.wav')
        shutil.copy(org_file, temp_file)
        try:
            with open(temp_file, 'rb') as f:
                pass
        except OSError:
            raise unittest.SkipTest('the path cannot be opened')
        try:
            sound = mixer.Sound(temp_file)
            del sound
        finally:
            os.remove(temp_file)

    @unittest.skipIf(os.environ.get('SDL_AUDIODRIVER') == 'disk', 'this test fails without real sound card')
    def test_array_keyword(self):
        try:
            from numpy import array, arange, zeros, int8, uint8, int16, uint16, int32, uint32
        except ImportError:
            self.skipTest('requires numpy')
        freq = 22050
        format_list = [-8, 8, -16, 16]
        channels_list = [1, 2]
        a_lists = {f: [] for f in format_list}
        a32u_mono = arange(0, 256, 1, uint32)
        a16u_mono = a32u_mono.astype(uint16)
        a8u_mono = a32u_mono.astype(uint8)
        au_list_mono = [(1, a) for a in [a8u_mono, a16u_mono, a32u_mono]]
        for format in format_list:
            if format > 0:
                a_lists[format].extend(au_list_mono)
        a32s_mono = arange(-128, 128, 1, int32)
        a16s_mono = a32s_mono.astype(int16)
        a8s_mono = a32s_mono.astype(int8)
        as_list_mono = [(1, a) for a in [a8s_mono, a16s_mono, a32s_mono]]
        for format in format_list:
            if format < 0:
                a_lists[format].extend(as_list_mono)
        a32u_stereo = zeros([a32u_mono.shape[0], 2], uint32)
        a32u_stereo[:, 0] = a32u_mono
        a32u_stereo[:, 1] = 255 - a32u_mono
        a16u_stereo = a32u_stereo.astype(uint16)
        a8u_stereo = a32u_stereo.astype(uint8)
        au_list_stereo = [(2, a) for a in [a8u_stereo, a16u_stereo, a32u_stereo]]
        for format in format_list:
            if format > 0:
                a_lists[format].extend(au_list_stereo)
        a32s_stereo = zeros([a32s_mono.shape[0], 2], int32)
        a32s_stereo[:, 0] = a32s_mono
        a32s_stereo[:, 1] = -1 - a32s_mono
        a16s_stereo = a32s_stereo.astype(int16)
        a8s_stereo = a32s_stereo.astype(int8)
        as_list_stereo = [(2, a) for a in [a8s_stereo, a16s_stereo, a32s_stereo]]
        for format in format_list:
            if format < 0:
                a_lists[format].extend(as_list_stereo)
        for format in format_list:
            for channels in channels_list:
                try:
                    mixer.init(freq, format, channels)
                except pygame.error:
                    continue
                try:
                    __, f, c = mixer.get_init()
                    if f != format or c != channels:
                        continue
                    for c, a in a_lists[format]:
                        self._test_array_argument(format, a, c == channels)
                finally:
                    mixer.quit()

    def _test_array_argument(self, format, a, test_pass):
        from numpy import array, all as all_
        try:
            snd = mixer.Sound(array=a)
        except ValueError:
            if not test_pass:
                return
            self.fail('Raised ValueError: Format %i, dtype %s' % (format, a.dtype))
        if not test_pass:
            self.fail('Did not raise ValueError: Format %i, dtype %s' % (format, a.dtype))
        a2 = array(snd)
        a3 = a.astype(a2.dtype)
        lshift = abs(format) - 8 * a.itemsize
        if lshift >= 0:
            a3 <<= lshift
        self.assertTrue(all_(a2 == a3), 'Format %i, dtype %s' % (format, a.dtype))

    def _test_array_interface_fail(self, a):
        self.assertRaises(ValueError, mixer.Sound, array=a)

    def test_array_interface(self):
        mixer.init(22050, -16, 1, allowedchanges=0)
        snd = mixer.Sound(buffer=b'\x00\x7f' * 20)
        d = snd.__array_interface__
        self.assertTrue(isinstance(d, dict))
        if pygame.get_sdl_byteorder() == pygame.LIL_ENDIAN:
            typestr = '<i2'
        else:
            typestr = '>i2'
        self.assertEqual(d['typestr'], typestr)
        self.assertEqual(d['shape'], (20,))
        self.assertEqual(d['strides'], (2,))
        self.assertEqual(d['data'], (snd._samples_address, False))

    @unittest.skipIf(not pygame.HAVE_NEWBUF, 'newbuf not implemented')
    @unittest.skipIf(IS_PYPY, 'pypy no likey')
    def test_newbuf__one_channel(self):
        mixer.init(22050, -16, 1)
        self._NEWBUF_export_check()

    @unittest.skipIf(not pygame.HAVE_NEWBUF, 'newbuf not implemented')
    @unittest.skipIf(IS_PYPY, 'pypy no likey')
    def test_newbuf__twho_channel(self):
        mixer.init(22050, -16, 2)
        self._NEWBUF_export_check()

    def _NEWBUF_export_check(self):
        freq, fmt, channels = mixer.get_init()
        ndim = 1 if channels == 1 else 2
        itemsize = abs(fmt) // 8
        formats = {8: 'B', -8: 'b', 16: '=H', -16: '=h', 32: '=I', -32: '=i', 64: '=Q', -64: '=q'}
        format = formats[fmt]
        from pygame.tests.test_utils import buftools
        Exporter = buftools.Exporter
        Importer = buftools.Importer
        is_lil_endian = pygame.get_sdl_byteorder() == pygame.LIL_ENDIAN
        fsys, frev = ('<', '>') if is_lil_endian else ('>', '<')
        shape = (10, channels)[:ndim]
        strides = (channels * itemsize, itemsize)[2 - ndim:]
        exp = Exporter(shape, format=frev + 'i')
        snd = mixer.Sound(array=exp)
        buflen = len(exp) * itemsize * channels
        imp = Importer(snd, buftools.PyBUF_SIMPLE)
        self.assertEqual(imp.ndim, 0)
        self.assertTrue(imp.format is None)
        self.assertEqual(imp.len, buflen)
        self.assertEqual(imp.itemsize, itemsize)
        self.assertTrue(imp.shape is None)
        self.assertTrue(imp.strides is None)
        self.assertTrue(imp.suboffsets is None)
        self.assertFalse(imp.readonly)
        self.assertEqual(imp.buf, snd._samples_address)
        imp = Importer(snd, buftools.PyBUF_WRITABLE)
        self.assertEqual(imp.ndim, 0)
        self.assertTrue(imp.format is None)
        self.assertEqual(imp.len, buflen)
        self.assertEqual(imp.itemsize, itemsize)
        self.assertTrue(imp.shape is None)
        self.assertTrue(imp.strides is None)
        self.assertTrue(imp.suboffsets is None)
        self.assertFalse(imp.readonly)
        self.assertEqual(imp.buf, snd._samples_address)
        imp = Importer(snd, buftools.PyBUF_FORMAT)
        self.assertEqual(imp.ndim, 0)
        self.assertEqual(imp.format, format)
        self.assertEqual(imp.len, buflen)
        self.assertEqual(imp.itemsize, itemsize)
        self.assertTrue(imp.shape is None)
        self.assertTrue(imp.strides is None)
        self.assertTrue(imp.suboffsets is None)
        self.assertFalse(imp.readonly)
        self.assertEqual(imp.buf, snd._samples_address)
        imp = Importer(snd, buftools.PyBUF_ND)
        self.assertEqual(imp.ndim, ndim)
        self.assertTrue(imp.format is None)
        self.assertEqual(imp.len, buflen)
        self.assertEqual(imp.itemsize, itemsize)
        self.assertEqual(imp.shape, shape)
        self.assertTrue(imp.strides is None)
        self.assertTrue(imp.suboffsets is None)
        self.assertFalse(imp.readonly)
        self.assertEqual(imp.buf, snd._samples_address)
        imp = Importer(snd, buftools.PyBUF_STRIDES)
        self.assertEqual(imp.ndim, ndim)
        self.assertTrue(imp.format is None)
        self.assertEqual(imp.len, buflen)
        self.assertEqual(imp.itemsize, itemsize)
        self.assertEqual(imp.shape, shape)
        self.assertEqual(imp.strides, strides)
        self.assertTrue(imp.suboffsets is None)
        self.assertFalse(imp.readonly)
        self.assertEqual(imp.buf, snd._samples_address)
        imp = Importer(snd, buftools.PyBUF_FULL_RO)
        self.assertEqual(imp.ndim, ndim)
        self.assertEqual(imp.format, format)
        self.assertEqual(imp.len, buflen)
        self.assertEqual(imp.itemsize, 2)
        self.assertEqual(imp.shape, shape)
        self.assertEqual(imp.strides, strides)
        self.assertTrue(imp.suboffsets is None)
        self.assertFalse(imp.readonly)
        self.assertEqual(imp.buf, snd._samples_address)
        imp = Importer(snd, buftools.PyBUF_FULL_RO)
        self.assertEqual(imp.ndim, ndim)
        self.assertEqual(imp.format, format)
        self.assertEqual(imp.len, buflen)
        self.assertEqual(imp.itemsize, itemsize)
        self.assertEqual(imp.shape, exp.shape)
        self.assertEqual(imp.strides, strides)
        self.assertTrue(imp.suboffsets is None)
        self.assertFalse(imp.readonly)
        self.assertEqual(imp.buf, snd._samples_address)
        imp = Importer(snd, buftools.PyBUF_C_CONTIGUOUS)
        self.assertEqual(imp.ndim, ndim)
        self.assertTrue(imp.format is None)
        self.assertEqual(imp.strides, strides)
        imp = Importer(snd, buftools.PyBUF_ANY_CONTIGUOUS)
        self.assertEqual(imp.ndim, ndim)
        self.assertTrue(imp.format is None)
        self.assertEqual(imp.strides, strides)
        if ndim == 1:
            imp = Importer(snd, buftools.PyBUF_F_CONTIGUOUS)
            self.assertEqual(imp.ndim, 1)
            self.assertTrue(imp.format is None)
            self.assertEqual(imp.strides, strides)
        else:
            self.assertRaises(BufferError, Importer, snd, buftools.PyBUF_F_CONTIGUOUS)

    def test_fadeout(self):
        """Ensure pygame.mixer.fadeout() stops playback after fading out the sound."""
        if mixer.get_init() is None:
            mixer.init()
        sound = pygame.mixer.Sound(example_path('data/house_lo.wav'))
        channel = pygame.mixer.find_channel()
        channel.play(sound)
        fadeout_time = 200
        channel.fadeout(fadeout_time)
        pygame.time.wait(fadeout_time + 30)
        self.assertFalse(channel.get_busy())

    def test_find_channel(self):
        mixer.init()
        filename = example_path(os.path.join('data', 'house_lo.wav'))
        sound = mixer.Sound(file=filename)
        num_channels = mixer.get_num_channels()
        if num_channels > 0:
            found_channel = mixer.find_channel()
            self.assertIsNotNone(found_channel)
            channels = []
            for channel_id in range(0, num_channels):
                channel = mixer.Channel(channel_id)
                channel.play(sound)
                channels.append(channel)
            found_channel = mixer.find_channel()
            self.assertIsNone(found_channel)
            found_channel = mixer.find_channel(True)
            self.assertIsNotNone(found_channel)
            found_channel = mixer.find_channel(force=True)
            self.assertIsNotNone(found_channel)
            for channel in channels:
                channel.stop()
            found_channel = mixer.find_channel()
            self.assertIsNotNone(found_channel)

    @unittest.expectedFailure
    def test_pause(self):
        """Ensure pygame.mixer.pause() temporarily stops playback of all sound channels."""
        if mixer.get_init() is None:
            mixer.init()
        sound = mixer.Sound(example_path('data/house_lo.wav'))
        channel = mixer.find_channel()
        channel.play(sound)
        mixer.pause()
        self.assertFalse(channel.get_busy())
        mixer.unpause()
        self.assertTrue(channel.get_busy())

    def test_set_reserved(self):
        """Ensure pygame.mixer.set_reserved() reserves the given number of channels."""
        mixer.init()
        default_num_channels = mixer.get_num_channels()
        result = mixer.set_reserved(default_num_channels)
        self.assertEqual(result, default_num_channels)
        result = mixer.set_reserved(default_num_channels + 1)
        self.assertEqual(result, default_num_channels)
        result = mixer.set_reserved(0)
        self.assertEqual(result, 0)
        result = mixer.set_reserved(int(default_num_channels / 2))
        self.assertEqual(result, int(default_num_channels / 2))

    def test_stop(self):
        """Stops playback of all active sound channels."""
        if mixer.get_init() is None:
            mixer.init()
        sound = pygame.mixer.Sound(example_path('data/house_lo.wav'))
        channel = pygame.mixer.Channel(0)
        channel.play(sound)
        pygame.mixer.stop()
        for i in range(pygame.mixer.get_num_channels()):
            self.assertFalse(pygame.mixer.Channel(i).get_busy())

    def test_get_sdl_mixer_version(self):
        """Ensures get_sdl_mixer_version works correctly with no args."""
        expected_length = 3
        expected_type = tuple
        expected_item_type = int
        version = pygame.mixer.get_sdl_mixer_version()
        self.assertIsInstance(version, expected_type)
        self.assertEqual(len(version), expected_length)
        for item in version:
            self.assertIsInstance(item, expected_item_type)

    def test_get_sdl_mixer_version__args(self):
        """Ensures get_sdl_mixer_version works correctly using args."""
        expected_length = 3
        expected_type = tuple
        expected_item_type = int
        for value in (True, False):
            version = pygame.mixer.get_sdl_mixer_version(value)
            self.assertIsInstance(version, expected_type)
            self.assertEqual(len(version), expected_length)
            for item in version:
                self.assertIsInstance(item, expected_item_type)

    def test_get_sdl_mixer_version__kwargs(self):
        """Ensures get_sdl_mixer_version works correctly using kwargs."""
        expected_length = 3
        expected_type = tuple
        expected_item_type = int
        for value in (True, False):
            version = pygame.mixer.get_sdl_mixer_version(linked=value)
            self.assertIsInstance(version, expected_type)
            self.assertEqual(len(version), expected_length)
            for item in version:
                self.assertIsInstance(item, expected_item_type)

    def test_get_sdl_mixer_version__invalid_args_kwargs(self):
        """Ensures get_sdl_mixer_version handles invalid args and kwargs."""
        invalid_bool = InvalidBool()
        with self.assertRaises(TypeError):
            version = pygame.mixer.get_sdl_mixer_version(invalid_bool)
        with self.assertRaises(TypeError):
            version = pygame.mixer.get_sdl_mixer_version(linked=invalid_bool)

    def test_get_sdl_mixer_version__linked_equals_compiled(self):
        """Ensures get_sdl_mixer_version's linked/compiled versions are equal."""
        linked_version = pygame.mixer.get_sdl_mixer_version(linked=True)
        complied_version = pygame.mixer.get_sdl_mixer_version(linked=False)
        self.assertTupleEqual(linked_version, complied_version)