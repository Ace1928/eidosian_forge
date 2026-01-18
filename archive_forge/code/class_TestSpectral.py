from numpy.testing import (assert_allclose, assert_almost_equal,
import numpy as np
import pytest
from matplotlib import mlab
@pytest.mark.parametrize('iscomplex', [False, True], ids=['real', 'complex'], scope='class')
@pytest.mark.parametrize('sides', ['onesided', 'twosided', 'default'], scope='class')
@pytest.mark.parametrize('fstims,len_x,NFFT_density,nover_density,pad_to_density,pad_to_spectrum', [([], None, -1, -1, -1, -1), ([4], None, -1, -1, -1, -1), ([4, 5, 10], None, -1, -1, -1, -1), ([], None, None, -1, -1, None), ([], None, -1, -1, None, None), ([], None, None, -1, None, None), ([], 1024, 512, -1, -1, 128), ([], 256, -1, -1, 33, 257), ([], 255, 33, -1, -1, None), ([], 256, 128, -1, 256, 256), ([], None, -1, 32, -1, -1)], ids=['nosig', 'Fs4', 'FsAll', 'nosig_noNFFT', 'nosig_nopad_to', 'nosig_noNFFT_no_pad_to', 'nosig_trim', 'nosig_odd', 'nosig_oddlen', 'nosig_stretch', 'nosig_overlap'], scope='class')
class TestSpectral:

    @pytest.fixture(scope='class', autouse=True)
    def stim(self, request, fstims, iscomplex, sides, len_x, NFFT_density, nover_density, pad_to_density, pad_to_spectrum):
        Fs = 100.0
        x = np.arange(0, 10, 1 / Fs)
        if len_x is not None:
            x = x[:len_x]
        fstims = [Fs / fstim for fstim in fstims]
        if NFFT_density is None:
            NFFT_density_real = 256
        elif NFFT_density < 0:
            NFFT_density_real = NFFT_density = 100
        else:
            NFFT_density_real = NFFT_density
        if nover_density is None:
            nover_density_real = 0
        elif nover_density < 0:
            nover_density_real = nover_density = NFFT_density_real // 2
        else:
            nover_density_real = nover_density
        if pad_to_density is None:
            pad_to_density_real = NFFT_density_real
        elif pad_to_density < 0:
            pad_to_density = int(2 ** np.ceil(np.log2(NFFT_density_real)))
            pad_to_density_real = pad_to_density
        else:
            pad_to_density_real = pad_to_density
        if pad_to_spectrum is None:
            pad_to_spectrum_real = len(x)
        elif pad_to_spectrum < 0:
            pad_to_spectrum_real = pad_to_spectrum = len(x)
        else:
            pad_to_spectrum_real = pad_to_spectrum
        if pad_to_spectrum is None:
            NFFT_spectrum_real = NFFT_spectrum = pad_to_spectrum_real
        else:
            NFFT_spectrum_real = NFFT_spectrum = len(x)
        nover_spectrum = 0
        NFFT_specgram = NFFT_density
        nover_specgram = nover_density
        pad_to_specgram = pad_to_density
        NFFT_specgram_real = NFFT_density_real
        nover_specgram_real = nover_density_real
        if sides == 'onesided' or (sides == 'default' and (not iscomplex)):
            if pad_to_density_real % 2:
                freqs_density = np.linspace(0, Fs / 2, num=pad_to_density_real, endpoint=False)[::2]
            else:
                freqs_density = np.linspace(0, Fs / 2, num=pad_to_density_real // 2 + 1)
            if pad_to_spectrum_real % 2:
                freqs_spectrum = np.linspace(0, Fs / 2, num=pad_to_spectrum_real, endpoint=False)[::2]
            else:
                freqs_spectrum = np.linspace(0, Fs / 2, num=pad_to_spectrum_real // 2 + 1)
        else:
            if pad_to_density_real % 2:
                freqs_density = np.linspace(-Fs / 2, Fs / 2, num=2 * pad_to_density_real, endpoint=False)[1::2]
            else:
                freqs_density = np.linspace(-Fs / 2, Fs / 2, num=pad_to_density_real, endpoint=False)
            if pad_to_spectrum_real % 2:
                freqs_spectrum = np.linspace(-Fs / 2, Fs / 2, num=2 * pad_to_spectrum_real, endpoint=False)[1::2]
            else:
                freqs_spectrum = np.linspace(-Fs / 2, Fs / 2, num=pad_to_spectrum_real, endpoint=False)
        freqs_specgram = freqs_density
        t_start = NFFT_specgram_real // 2
        t_stop = len(x) - NFFT_specgram_real // 2 + 1
        t_step = NFFT_specgram_real - nover_specgram_real
        t_specgram = x[t_start:t_stop:t_step]
        if NFFT_specgram_real % 2:
            t_specgram += 1 / Fs / 2
        if len(t_specgram) == 0:
            t_specgram = np.array([NFFT_specgram_real / (2 * Fs)])
        t_spectrum = np.array([NFFT_spectrum_real / (2 * Fs)])
        t_density = t_specgram
        y = np.zeros_like(x)
        for i, fstim in enumerate(fstims):
            y += np.sin(fstim * x * np.pi * 2) * 10 ** i
        if iscomplex:
            y = y.astype('complex')
        cls = request.cls
        cls.Fs = Fs
        cls.sides = sides
        cls.fstims = fstims
        cls.NFFT_density = NFFT_density
        cls.nover_density = nover_density
        cls.pad_to_density = pad_to_density
        cls.NFFT_spectrum = NFFT_spectrum
        cls.nover_spectrum = nover_spectrum
        cls.pad_to_spectrum = pad_to_spectrum
        cls.NFFT_specgram = NFFT_specgram
        cls.nover_specgram = nover_specgram
        cls.pad_to_specgram = pad_to_specgram
        cls.t_specgram = t_specgram
        cls.t_density = t_density
        cls.t_spectrum = t_spectrum
        cls.y = y
        cls.freqs_density = freqs_density
        cls.freqs_spectrum = freqs_spectrum
        cls.freqs_specgram = freqs_specgram
        cls.NFFT_density_real = NFFT_density_real

    def check_freqs(self, vals, targfreqs, resfreqs, fstims):
        assert resfreqs.argmin() == 0
        assert resfreqs.argmax() == len(resfreqs) - 1
        assert_allclose(resfreqs, targfreqs, atol=1e-06)
        for fstim in fstims:
            i = np.abs(resfreqs - fstim).argmin()
            assert vals[i] > vals[i + 2]
            assert vals[i] > vals[i - 2]

    def check_maxfreq(self, spec, fsp, fstims):
        if len(fstims) == 0:
            return
        if fsp.min() < 0:
            fspa = np.abs(fsp)
            zeroind = fspa.argmin()
            self.check_maxfreq(spec[:zeroind], fspa[:zeroind], fstims)
            self.check_maxfreq(spec[zeroind:], fspa[zeroind:], fstims)
            return
        fstimst = fstims[:]
        spect = spec.copy()
        while fstimst:
            maxind = spect.argmax()
            maxfreq = fsp[maxind]
            assert_almost_equal(maxfreq, fstimst[-1])
            del fstimst[-1]
            spect[maxind - 5:maxind + 5] = 0

    def test_spectral_helper_raises(self):
        for kwargs in [{'y': self.y + 1, 'mode': 'complex'}, {'y': self.y + 1, 'mode': 'magnitude'}, {'y': self.y + 1, 'mode': 'angle'}, {'y': self.y + 1, 'mode': 'phase'}, {'mode': 'spam'}, {'y': self.y, 'sides': 'eggs'}, {'y': self.y, 'NFFT': 10, 'noverlap': 20}, {'NFFT': 10, 'noverlap': 10}, {'y': self.y, 'NFFT': 10, 'window': np.ones(9)}]:
            with pytest.raises(ValueError):
                mlab._spectral_helper(x=self.y, **kwargs)

    @pytest.mark.parametrize('mode', ['default', 'psd'])
    def test_single_spectrum_helper_unsupported_modes(self, mode):
        with pytest.raises(ValueError):
            mlab._single_spectrum_helper(x=self.y, mode=mode)

    @pytest.mark.parametrize('mode, case', [('psd', 'density'), ('magnitude', 'specgram'), ('magnitude', 'spectrum')])
    def test_spectral_helper_psd(self, mode, case):
        freqs = getattr(self, f'freqs_{case}')
        spec, fsp, t = mlab._spectral_helper(x=self.y, y=self.y, NFFT=getattr(self, f'NFFT_{case}'), Fs=self.Fs, noverlap=getattr(self, f'nover_{case}'), pad_to=getattr(self, f'pad_to_{case}'), sides=self.sides, mode=mode)
        assert_allclose(fsp, freqs, atol=1e-06)
        assert_allclose(t, getattr(self, f't_{case}'), atol=1e-06)
        assert spec.shape[0] == freqs.shape[0]
        assert spec.shape[1] == getattr(self, f't_{case}').shape[0]

    def test_csd(self):
        freqs = self.freqs_density
        spec, fsp = mlab.csd(x=self.y, y=self.y + 1, NFFT=self.NFFT_density, Fs=self.Fs, noverlap=self.nover_density, pad_to=self.pad_to_density, sides=self.sides)
        assert_allclose(fsp, freqs, atol=1e-06)
        assert spec.shape == freqs.shape

    def test_csd_padding(self):
        """Test zero padding of csd()."""
        if self.NFFT_density is None:
            return
        sargs = dict(x=self.y, y=self.y + 1, Fs=self.Fs, window=mlab.window_none, sides=self.sides)
        spec0, _ = mlab.csd(NFFT=self.NFFT_density, **sargs)
        spec1, _ = mlab.csd(NFFT=self.NFFT_density * 2, **sargs)
        assert_almost_equal(np.sum(np.conjugate(spec0) * spec0).real, np.sum(np.conjugate(spec1 / 2) * spec1 / 2).real)

    def test_psd(self):
        freqs = self.freqs_density
        spec, fsp = mlab.psd(x=self.y, NFFT=self.NFFT_density, Fs=self.Fs, noverlap=self.nover_density, pad_to=self.pad_to_density, sides=self.sides)
        assert spec.shape == freqs.shape
        self.check_freqs(spec, freqs, fsp, self.fstims)

    @pytest.mark.parametrize('make_data, detrend', [(np.zeros, mlab.detrend_mean), (np.zeros, 'mean'), (np.arange, mlab.detrend_linear), (np.arange, 'linear')])
    def test_psd_detrend(self, make_data, detrend):
        if self.NFFT_density is None:
            return
        ydata = make_data(self.NFFT_density)
        ydata1 = ydata + 5
        ydata2 = ydata + 3.3
        ydata = np.vstack([ydata1, ydata2])
        ydata = np.tile(ydata, (20, 1))
        ydatab = ydata.T.flatten()
        ydata = ydata.flatten()
        ycontrol = np.zeros_like(ydata)
        spec_g, fsp_g = mlab.psd(x=ydata, NFFT=self.NFFT_density, Fs=self.Fs, noverlap=0, sides=self.sides, detrend=detrend)
        spec_b, fsp_b = mlab.psd(x=ydatab, NFFT=self.NFFT_density, Fs=self.Fs, noverlap=0, sides=self.sides, detrend=detrend)
        spec_c, fsp_c = mlab.psd(x=ycontrol, NFFT=self.NFFT_density, Fs=self.Fs, noverlap=0, sides=self.sides)
        assert_array_equal(fsp_g, fsp_c)
        assert_array_equal(fsp_b, fsp_c)
        assert_allclose(spec_g, spec_c, atol=1e-08)
        with pytest.raises(AssertionError):
            assert_allclose(spec_b, spec_c, atol=1e-08)

    def test_psd_window_hanning(self):
        if self.NFFT_density is None:
            return
        ydata = np.arange(self.NFFT_density)
        ydata1 = ydata + 5
        ydata2 = ydata + 3.3
        windowVals = mlab.window_hanning(np.ones_like(ydata1))
        ycontrol1 = ydata1 * windowVals
        ycontrol2 = mlab.window_hanning(ydata2)
        ydata = np.vstack([ydata1, ydata2])
        ycontrol = np.vstack([ycontrol1, ycontrol2])
        ydata = np.tile(ydata, (20, 1))
        ycontrol = np.tile(ycontrol, (20, 1))
        ydatab = ydata.T.flatten()
        ydataf = ydata.flatten()
        ycontrol = ycontrol.flatten()
        spec_g, fsp_g = mlab.psd(x=ydataf, NFFT=self.NFFT_density, Fs=self.Fs, noverlap=0, sides=self.sides, window=mlab.window_hanning)
        spec_b, fsp_b = mlab.psd(x=ydatab, NFFT=self.NFFT_density, Fs=self.Fs, noverlap=0, sides=self.sides, window=mlab.window_hanning)
        spec_c, fsp_c = mlab.psd(x=ycontrol, NFFT=self.NFFT_density, Fs=self.Fs, noverlap=0, sides=self.sides, window=mlab.window_none)
        spec_c *= len(ycontrol1) / (windowVals ** 2).sum()
        assert_array_equal(fsp_g, fsp_c)
        assert_array_equal(fsp_b, fsp_c)
        assert_allclose(spec_g, spec_c, atol=1e-08)
        with pytest.raises(AssertionError):
            assert_allclose(spec_b, spec_c, atol=1e-08)

    def test_psd_window_hanning_detrend_linear(self):
        if self.NFFT_density is None:
            return
        ydata = np.arange(self.NFFT_density)
        ycontrol = np.zeros(self.NFFT_density)
        ydata1 = ydata + 5
        ydata2 = ydata + 3.3
        ycontrol1 = ycontrol
        ycontrol2 = ycontrol
        windowVals = mlab.window_hanning(np.ones_like(ycontrol1))
        ycontrol1 = ycontrol1 * windowVals
        ycontrol2 = mlab.window_hanning(ycontrol2)
        ydata = np.vstack([ydata1, ydata2])
        ycontrol = np.vstack([ycontrol1, ycontrol2])
        ydata = np.tile(ydata, (20, 1))
        ycontrol = np.tile(ycontrol, (20, 1))
        ydatab = ydata.T.flatten()
        ydataf = ydata.flatten()
        ycontrol = ycontrol.flatten()
        spec_g, fsp_g = mlab.psd(x=ydataf, NFFT=self.NFFT_density, Fs=self.Fs, noverlap=0, sides=self.sides, detrend=mlab.detrend_linear, window=mlab.window_hanning)
        spec_b, fsp_b = mlab.psd(x=ydatab, NFFT=self.NFFT_density, Fs=self.Fs, noverlap=0, sides=self.sides, detrend=mlab.detrend_linear, window=mlab.window_hanning)
        spec_c, fsp_c = mlab.psd(x=ycontrol, NFFT=self.NFFT_density, Fs=self.Fs, noverlap=0, sides=self.sides, window=mlab.window_none)
        spec_c *= len(ycontrol1) / (windowVals ** 2).sum()
        assert_array_equal(fsp_g, fsp_c)
        assert_array_equal(fsp_b, fsp_c)
        assert_allclose(spec_g, spec_c, atol=1e-08)
        with pytest.raises(AssertionError):
            assert_allclose(spec_b, spec_c, atol=1e-08)

    def test_psd_window_flattop(self):
        a = [0.21557895, 0.41663158, 0.277263158, 0.083578947, 0.006947368]
        fac = np.linspace(-np.pi, np.pi, self.NFFT_density_real)
        win = np.zeros(self.NFFT_density_real)
        for k in range(len(a)):
            win += a[k] * np.cos(k * fac)
        spec, fsp = mlab.psd(x=self.y, NFFT=self.NFFT_density, Fs=self.Fs, noverlap=0, sides=self.sides, window=win, scale_by_freq=False)
        spec_a, fsp_a = mlab.psd(x=self.y, NFFT=self.NFFT_density, Fs=self.Fs, noverlap=0, sides=self.sides, window=win)
        assert_allclose(spec * win.sum() ** 2, spec_a * self.Fs * (win ** 2).sum(), atol=1e-08)

    def test_psd_windowarray(self):
        freqs = self.freqs_density
        spec, fsp = mlab.psd(x=self.y, NFFT=self.NFFT_density, Fs=self.Fs, noverlap=self.nover_density, pad_to=self.pad_to_density, sides=self.sides, window=np.ones(self.NFFT_density_real))
        assert_allclose(fsp, freqs, atol=1e-06)
        assert spec.shape == freqs.shape

    def test_psd_windowarray_scale_by_freq(self):
        win = mlab.window_hanning(np.ones(self.NFFT_density_real))
        spec, fsp = mlab.psd(x=self.y, NFFT=self.NFFT_density, Fs=self.Fs, noverlap=self.nover_density, pad_to=self.pad_to_density, sides=self.sides, window=mlab.window_hanning)
        spec_s, fsp_s = mlab.psd(x=self.y, NFFT=self.NFFT_density, Fs=self.Fs, noverlap=self.nover_density, pad_to=self.pad_to_density, sides=self.sides, window=mlab.window_hanning, scale_by_freq=True)
        spec_n, fsp_n = mlab.psd(x=self.y, NFFT=self.NFFT_density, Fs=self.Fs, noverlap=self.nover_density, pad_to=self.pad_to_density, sides=self.sides, window=mlab.window_hanning, scale_by_freq=False)
        assert_array_equal(fsp, fsp_s)
        assert_array_equal(fsp, fsp_n)
        assert_array_equal(spec, spec_s)
        assert_allclose(spec_s * (win ** 2).sum(), spec_n / self.Fs * win.sum() ** 2, atol=1e-08)

    @pytest.mark.parametrize('kind', ['complex', 'magnitude', 'angle', 'phase'])
    def test_spectrum(self, kind):
        freqs = self.freqs_spectrum
        spec, fsp = getattr(mlab, f'{kind}_spectrum')(x=self.y, Fs=self.Fs, sides=self.sides, pad_to=self.pad_to_spectrum)
        assert_allclose(fsp, freqs, atol=1e-06)
        assert spec.shape == freqs.shape
        if kind == 'magnitude':
            self.check_maxfreq(spec, fsp, self.fstims)
            self.check_freqs(spec, freqs, fsp, self.fstims)

    @pytest.mark.parametrize('kwargs', [{}, {'mode': 'default'}, {'mode': 'psd'}, {'mode': 'magnitude'}, {'mode': 'complex'}, {'mode': 'angle'}, {'mode': 'phase'}])
    def test_specgram(self, kwargs):
        freqs = self.freqs_specgram
        spec, fsp, t = mlab.specgram(x=self.y, NFFT=self.NFFT_specgram, Fs=self.Fs, noverlap=self.nover_specgram, pad_to=self.pad_to_specgram, sides=self.sides, **kwargs)
        if kwargs.get('mode') == 'complex':
            spec = np.abs(spec)
        specm = np.mean(spec, axis=1)
        assert_allclose(fsp, freqs, atol=1e-06)
        assert_allclose(t, self.t_specgram, atol=1e-06)
        assert spec.shape[0] == freqs.shape[0]
        assert spec.shape[1] == self.t_specgram.shape[0]
        if kwargs.get('mode') not in ['complex', 'angle', 'phase']:
            if np.abs(spec.max()) != 0:
                assert_allclose(np.diff(spec, axis=1).max() / np.abs(spec.max()), 0, atol=0.01)
        if kwargs.get('mode') not in ['angle', 'phase']:
            self.check_freqs(specm, freqs, fsp, self.fstims)

    def test_specgram_warn_only1seg(self):
        """Warning should be raised if len(x) <= NFFT."""
        with pytest.warns(UserWarning, match='Only one segment is calculated'):
            mlab.specgram(x=self.y, NFFT=len(self.y), Fs=self.Fs)

    def test_psd_csd_equal(self):
        Pxx, freqsxx = mlab.psd(x=self.y, NFFT=self.NFFT_density, Fs=self.Fs, noverlap=self.nover_density, pad_to=self.pad_to_density, sides=self.sides)
        Pxy, freqsxy = mlab.csd(x=self.y, y=self.y, NFFT=self.NFFT_density, Fs=self.Fs, noverlap=self.nover_density, pad_to=self.pad_to_density, sides=self.sides)
        assert_array_almost_equal_nulp(Pxx, Pxy)
        assert_array_equal(freqsxx, freqsxy)

    @pytest.mark.parametrize('mode', ['default', 'psd'])
    def test_specgram_auto_default_psd_equal(self, mode):
        """
        Test that mlab.specgram without mode and with mode 'default' and 'psd'
        are all the same.
        """
        speca, freqspeca, ta = mlab.specgram(x=self.y, NFFT=self.NFFT_specgram, Fs=self.Fs, noverlap=self.nover_specgram, pad_to=self.pad_to_specgram, sides=self.sides)
        specb, freqspecb, tb = mlab.specgram(x=self.y, NFFT=self.NFFT_specgram, Fs=self.Fs, noverlap=self.nover_specgram, pad_to=self.pad_to_specgram, sides=self.sides, mode=mode)
        assert_array_equal(speca, specb)
        assert_array_equal(freqspeca, freqspecb)
        assert_array_equal(ta, tb)

    @pytest.mark.parametrize('mode, conv', [('magnitude', np.abs), ('angle', np.angle), ('phase', lambda x: np.unwrap(np.angle(x), axis=0))])
    def test_specgram_complex_equivalent(self, mode, conv):
        specc, freqspecc, tc = mlab.specgram(x=self.y, NFFT=self.NFFT_specgram, Fs=self.Fs, noverlap=self.nover_specgram, pad_to=self.pad_to_specgram, sides=self.sides, mode='complex')
        specm, freqspecm, tm = mlab.specgram(x=self.y, NFFT=self.NFFT_specgram, Fs=self.Fs, noverlap=self.nover_specgram, pad_to=self.pad_to_specgram, sides=self.sides, mode=mode)
        assert_array_equal(freqspecc, freqspecm)
        assert_array_equal(tc, tm)
        assert_allclose(conv(specc), specm, atol=1e-06)

    def test_psd_windowarray_equal(self):
        win = mlab.window_hanning(np.ones(self.NFFT_density_real))
        speca, fspa = mlab.psd(x=self.y, NFFT=self.NFFT_density, Fs=self.Fs, noverlap=self.nover_density, pad_to=self.pad_to_density, sides=self.sides, window=win)
        specb, fspb = mlab.psd(x=self.y, NFFT=self.NFFT_density, Fs=self.Fs, noverlap=self.nover_density, pad_to=self.pad_to_density, sides=self.sides)
        assert_array_equal(fspa, fspb)
        assert_allclose(speca, specb, atol=1e-08)