import numpy as np
import scipy.misc as sc
import scipy.signal
import scipy.io
class Steerable:

    def __init__(self, height=4, order=4, twidth=1):
        """
		height is the total height, including highpass and lowpass
		"""
        self.nbands = np.round(order)
        self.nbands = np.double(self.nbands)
        self.height = height
        self.twidth = twidth

    def steerAngle(self, im, angle):
        pass

    def buildSFpyr(self, im):
        M, N = im.shape[:2]
        log_rad, angle = self.base(M, N)
        Xrcos, Yrcos = self.rcosFn(1, -0.5)
        Yrcos = np.sqrt(Yrcos)
        YIrcos = np.sqrt(1 - Yrcos * Yrcos)
        lo0mask = self.pointOp(log_rad, YIrcos, Xrcos)
        hi0mask = self.pointOp(log_rad, Yrcos, Xrcos)
        imdft = np.fft.fftshift(np.fft.fft2(im))
        lo0dft = imdft * lo0mask
        coeff = self.buildSFpyrlevs(lo0dft, log_rad, angle, Xrcos, Yrcos, self.height - 1)
        hi0dft = imdft * hi0mask
        hi0 = np.fft.ifft2(np.fft.ifftshift(hi0dft))
        coeff.insert(0, hi0.real)
        return coeff

    def buildSFpyrlevs(self, lodft, log_rad, angle, Xrcos, Yrcos, ht):
        if ht <= 1:
            lo0 = np.fft.ifft2(np.fft.ifftshift(lodft))
            coeff = [lo0.real]
        else:
            Xrcos = Xrcos - np.log2(2)
            himask = self.pointOp(log_rad, Yrcos, Xrcos)
            lutsize = 1024
            Xcosn = np.pi * np.array(range(-(2 * lutsize + 1), lutsize + 2)) / lutsize
            order = self.nbands - 1
            const = 2 ** (2 * order) * sc.factorial(order) ** 2 / (self.nbands * sc.factorial(2 * order))
            Ycosn = np.sqrt(const) * np.cos(Xcosn) ** order
            M, N = np.shape(lodft)
            orients = np.zeros((int(self.nbands), M, N))
            for b in range(int(self.nbands)):
                anglemask = self.pointOp(angle, Ycosn, Xcosn + np.pi * b / self.nbands).astype(np.complex)
                banddft = np.complex(0, -1) ** order * lodft
                banddft *= anglemask
                banddft *= himask
                orients[b, :, :] = np.fft.ifft2(np.fft.ifftshift(banddft)).real
            dims = np.array(lodft.shape)
            lostart = np.ceil((dims + 0.5) / 2) - np.ceil((np.ceil((dims - 0.5) / 2) + 0.5) / 2)
            loend = lostart + np.ceil((dims - 0.5) / 2)
            lostart = lostart.astype(int)
            loend = loend.astype(int)
            log_rad = log_rad[lostart[0]:loend[0], lostart[1]:loend[1]]
            angle = angle[lostart[0]:loend[0], lostart[1]:loend[1]]
            lodft = lodft[lostart[0]:loend[0], lostart[1]:loend[1]]
            YIrcos = np.abs(np.sqrt(1 - Yrcos * Yrcos))
            lomask = self.pointOp(log_rad, YIrcos, Xrcos)
            lodft = lomask * lodft
            coeff = self.buildSFpyrlevs(lodft, log_rad, angle, Xrcos, Yrcos, ht - 1)
            coeff.insert(0, orients)
        return coeff

    def reconSFPyrLevs(self, coeff, log_rad, Xrcos, Yrcos, angle):
        if len(coeff) == 1:
            return np.fft.fftshift(np.fft.fft2(coeff[0]))
        else:
            Xrcos = Xrcos - 1
            himask = self.pointOp(log_rad, Yrcos, Xrcos)
            lutsize = 1024
            Xcosn = np.pi * np.array(range(-(2 * lutsize + 1), lutsize + 2)) / lutsize
            order = self.nbands - 1
            const = np.power(2, 2 * order) * np.square(sc.factorial(order)) / (self.nbands * sc.factorial(2 * order))
            Ycosn = np.sqrt(const) * np.power(np.cos(Xcosn), order)
            orientdft = np.zeros(coeff[0][0].shape, 'complex')
            for b in range(int(self.nbands)):
                anglemask = self.pointOp(angle, Ycosn, Xcosn + np.pi * b / self.nbands)
                banddft = np.fft.fftshift(np.fft.fft2(coeff[0][b]))
                orientdft += np.complex(0, 1) ** order * banddft * anglemask * himask
            dims = np.array(coeff[0][0].shape)
            lostart = np.ceil((dims + 0.5) / 2) - np.ceil((np.ceil((dims - 0.5) / 2) + 0.5) / 2)
            loend = lostart + np.ceil((dims - 0.5) / 2)
            nlog_rad = log_rad[lostart[0]:loend[0], lostart[1]:loend[1]]
            nangle = angle[lostart[0]:loend[0], lostart[1]:loend[1]]
            YIrcos = np.sqrt(np.abs(1 - Yrcos * Yrcos))
            lomask = self.pointOp(nlog_rad, YIrcos, Xrcos)
            nresdft = self.reconSFPyrLevs(coeff[1:], nlog_rad, Xrcos, Yrcos, nangle)
            res = np.fft.fftshift(np.fft.fft2(nresdft))
            resdft = np.zeros(dims, 'complex')
            resdft[lostart[0]:loend[0], lostart[1]:loend[1]] = nresdft * lomask
            return resdft + orientdft

    def reconSFpyr(self, coeff):
        if self.nbands != len(coeff[1]):
            raise Exception('Unmatched number of orientations')
        M, N = coeff[0].shape
        log_rad, angle = self.base(M, N)
        Xrcos, Yrcos = self.rcosFn(1, -0.5)
        Yrcos = np.sqrt(Yrcos)
        YIrcos = np.sqrt(np.abs(1 - Yrcos * Yrcos))
        lo0mask = self.pointOp(log_rad, YIrcos, Xrcos)
        hi0mask = self.pointOp(log_rad, Yrcos, Xrcos)
        tempdft = self.reconSFPyrLevs(coeff[1:], log_rad, Xrcos, Yrcos, angle)
        hidft = np.fft.fftshift(np.fft.fft2(coeff[0]))
        outdft = tempdft * lo0mask + hidft * hi0mask
        return np.fft.ifft2(np.fft.ifftshift(outdft)).real

    def base(self, m, n):
        ctrm = np.ceil((m + 0.5) / 2).astype(int)
        ctrn = np.ceil((n + 0.5) / 2).astype(int)
        xv, yv = np.meshgrid(np.array(range(n)) + 1 - ctrn, np.array(range(m)) + 1 - ctrm)
        xv = xv.astype(np.double)
        yv = yv.astype(np.double)
        xv *= 2.0 / n
        yv *= 2.0 / m
        rad = np.sqrt(xv ** 2 + yv ** 2)
        rad[ctrm - 1, ctrn - 1] = rad[ctrm - 1, ctrn - 2]
        log_rad = np.log2(rad)
        angle = np.arctan2(yv, xv)
        return (log_rad, angle)

    def rcosFn(self, width, position):
        N = 256
        X = np.pi * np.array(range(-N - 1, 2))
        X /= 2.0 * N
        Y = np.cos(X) ** 2
        Y[0] = Y[1]
        Y[N + 2] = Y[N + 1]
        X = position + 2 * width / np.pi * (X + np.pi / 4)
        return (X, Y)

    def pointOp(self, im, Y, X):
        out = np.interp(im.flatten(), X, Y)
        return np.reshape(out, im.shape)

    def normalize(self, coef, height, order):
        filtsize = (3, 3)
        norm_bands = []
        for pyr_h in xrange(height - 2):
            inner_norm_bands = []
            sublevel = coef[pyr_h + 1]
            for cband in xrange(order):
                child = coef[0]
                parent = []
                w, h = np.shape(sublevel[cband])
                if pyr_h > 0:
                    child = scipy.misc.imresize(coef[pyr_h][cband], 50, interp='bilinear', mode='F')
                if pyr_h + 3 < height:
                    parent = scipy.misc.imresize(coef[pyr_h + 2][cband], 200, interp='bilinear', mode='F')
                    parent = parent[1:-1, 1:-1]
                    wp, hp = np.shape(parent)
                    if wp > w - 2:
                        parent = parent[:w - 2, :]
                    if hp > h - 2:
                        parent = parent[:, :h - 2]
                idx = np.hstack((np.arange(0, cband), np.arange(cband + 1, order)))
                if parent == []:
                    cov = np.array(np.hstack((rolling_window(sublevel[cband], filtsize).reshape(((w - 2) * (h - 2), 9)), sublevel.transpose(1, 2, 0)[1:-1, 1:-1, idx].reshape((w - 2) * (h - 2), order - 1))))
                else:
                    cov = np.array(np.hstack((rolling_window(sublevel[cband], filtsize).reshape(((w - 2) * (h - 2), 9)), sublevel.transpose(1, 2, 0)[1:-1, 1:-1, idx].reshape((w - 2) * (h - 2), order - 1), np.matrix(parent.reshape((w - 2) * (h - 2))).T)))
                _, _, cov_mat = mean_cov(cov)
                N = np.shape(cov_mat)[0]
                N = 10 - pyr_h
                eigval, eigvec = np.linalg.eig(cov_mat)
                Q = np.matrix(eigvec)
                xdiag = np.matrix(np.diag(np.maximum(eigval, 0)))
                cov_mat = Q * xdiag * Q.T
                cov_inv = np.linalg.pinv(cov_mat)
                z = np.sqrt(np.einsum('ij,ij->i', np.dot(cov, cov_inv), cov) / N)
                cov[:, 4] -= np.average(cov[:, 4])
                result = cov[:, 4] / z
                gb = 16 / 2 ** pyr_h
                result = result.reshape(w - 2, h - 2)[gb:-gb, gb:-gb]
                result -= np.average(result)
                inner_norm_bands.append(result)
            norm_bands.append(inner_norm_bands)
        norm_bands = np.array(norm_bands)
        return norm_bands