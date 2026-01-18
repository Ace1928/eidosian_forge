import copy
import os
import pickle
import warnings
import numpy as np
def _readData2(self, fd, meta, mmap=False, subset=None, **kwds):
    dynAxis = None
    frameSize = 1
    for i in range(len(meta['info'])):
        ax = meta['info'][i]
        if 'values_len' in ax:
            if ax['values_len'] == 'dynamic':
                if dynAxis is not None:
                    raise Exception('MetaArray has more than one dynamic axis! (this is not allowed)')
                dynAxis = i
            else:
                ax['values'] = np.frombuffer(fd.read(ax['values_len']), dtype=ax['values_type'])
                frameSize *= ax['values_len']
                del ax['values_len']
                del ax['values_type']
    self._info = meta['info']
    if not kwds.get('readAllData', True):
        return
    if dynAxis is None:
        if meta['type'] == 'object':
            if mmap:
                raise Exception('memmap not supported for arrays with dtype=object')
            subarr = pickle.loads(fd.read())
        elif mmap:
            subarr = np.memmap(fd, dtype=meta['type'], mode='r', shape=meta['shape'])
        else:
            subarr = np.frombuffer(fd.read(), dtype=meta['type'])
        subarr.shape = meta['shape']
    else:
        if mmap:
            raise Exception('memmap not supported for non-contiguous arrays. Use rewriteContiguous() to convert.')
        ax = meta['info'][dynAxis]
        xVals = []
        frames = []
        frameShape = list(meta['shape'])
        frameShape[dynAxis] = 1
        frameSize = np.prod(frameShape)
        n = 0
        while True:
            while True:
                line = fd.readline()
                if line != '\n':
                    break
            if line == '':
                break
            inf = eval(line)
            if meta['type'] == 'object':
                data = pickle.loads(fd.read(inf['len']))
            else:
                data = np.frombuffer(fd.read(inf['len']), dtype=meta['type'])
            if data.size != frameSize * inf['numFrames']:
                raise Exception('Wrong frame size in MetaArray file! (frame %d)' % n)
            shape = list(frameShape)
            shape[dynAxis] = inf['numFrames']
            data.shape = shape
            if subset is not None:
                dSlice = subset[dynAxis]
                if dSlice.start is None:
                    dStart = 0
                else:
                    dStart = max(0, dSlice.start - n)
                if dSlice.stop is None:
                    dStop = data.shape[dynAxis]
                else:
                    dStop = min(data.shape[dynAxis], dSlice.stop - n)
                newSubset = list(subset[:])
                newSubset[dynAxis] = slice(dStart, dStop)
                if dStop > dStart:
                    frames.append(data[tuple(newSubset)].copy())
            else:
                frames.append(data)
            n += inf['numFrames']
            if 'xVals' in inf:
                xVals.extend(inf['xVals'])
        subarr = np.concatenate(frames, axis=dynAxis)
        if len(xVals) > 0:
            ax['values'] = np.array(xVals, dtype=ax['values_type'])
        del ax['values_len']
        del ax['values_type']
    self._info = meta['info']
    self._data = subarr