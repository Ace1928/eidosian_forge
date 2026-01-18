from __future__ import annotations
import itertools
import logging
import re
import struct
import warnings
import zlib
from enum import IntEnum
from . import Image, ImageChops, ImageFile, ImagePalette, ImageSequence
from ._binary import i16be as i16
from ._binary import i32be as i32
from ._binary import o8
from ._binary import o16be as o16
from ._binary import o32be as o32
def _write_multiple_frames(im, fp, chunk, rawmode, default_image, append_images):
    duration = im.encoderinfo.get('duration', im.info.get('duration', 0))
    loop = im.encoderinfo.get('loop', im.info.get('loop', 0))
    disposal = im.encoderinfo.get('disposal', im.info.get('disposal', Disposal.OP_NONE))
    blend = im.encoderinfo.get('blend', im.info.get('blend', Blend.OP_SOURCE))
    if default_image:
        chain = itertools.chain(append_images)
    else:
        chain = itertools.chain([im], append_images)
    im_frames = []
    frame_count = 0
    for im_seq in chain:
        for im_frame in ImageSequence.Iterator(im_seq):
            if im_frame.mode == rawmode:
                im_frame = im_frame.copy()
            else:
                im_frame = im_frame.convert(rawmode)
            encoderinfo = im.encoderinfo.copy()
            if isinstance(duration, (list, tuple)):
                encoderinfo['duration'] = duration[frame_count]
            if isinstance(disposal, (list, tuple)):
                encoderinfo['disposal'] = disposal[frame_count]
            if isinstance(blend, (list, tuple)):
                encoderinfo['blend'] = blend[frame_count]
            frame_count += 1
            if im_frames:
                previous = im_frames[-1]
                prev_disposal = previous['encoderinfo'].get('disposal')
                prev_blend = previous['encoderinfo'].get('blend')
                if prev_disposal == Disposal.OP_PREVIOUS and len(im_frames) < 2:
                    prev_disposal = Disposal.OP_BACKGROUND
                if prev_disposal == Disposal.OP_BACKGROUND:
                    base_im = previous['im'].copy()
                    dispose = Image.core.fill('RGBA', im.size, (0, 0, 0, 0))
                    bbox = previous['bbox']
                    if bbox:
                        dispose = dispose.crop(bbox)
                    else:
                        bbox = (0, 0) + im.size
                    base_im.paste(dispose, bbox)
                elif prev_disposal == Disposal.OP_PREVIOUS:
                    base_im = im_frames[-2]['im']
                else:
                    base_im = previous['im']
                delta = ImageChops.subtract_modulo(im_frame.convert('RGBA'), base_im.convert('RGBA'))
                bbox = delta.getbbox(alpha_only=False)
                if not bbox and prev_disposal == encoderinfo.get('disposal') and (prev_blend == encoderinfo.get('blend')):
                    previous['encoderinfo']['duration'] += encoderinfo.get('duration', duration)
                    continue
            else:
                bbox = None
            if 'duration' not in encoderinfo:
                encoderinfo['duration'] = duration
            im_frames.append({'im': im_frame, 'bbox': bbox, 'encoderinfo': encoderinfo})
    if len(im_frames) == 1 and (not default_image):
        return im_frames[0]['im']
    chunk(fp, b'acTL', o32(len(im_frames)), o32(loop))
    if default_image:
        if im.mode != rawmode:
            im = im.convert(rawmode)
        ImageFile._save(im, _idat(fp, chunk), [('zip', (0, 0) + im.size, 0, rawmode)])
    seq_num = 0
    for frame, frame_data in enumerate(im_frames):
        im_frame = frame_data['im']
        if not frame_data['bbox']:
            bbox = (0, 0) + im_frame.size
        else:
            bbox = frame_data['bbox']
            im_frame = im_frame.crop(bbox)
        size = im_frame.size
        encoderinfo = frame_data['encoderinfo']
        frame_duration = int(round(encoderinfo['duration']))
        frame_disposal = encoderinfo.get('disposal', disposal)
        frame_blend = encoderinfo.get('blend', blend)
        chunk(fp, b'fcTL', o32(seq_num), o32(size[0]), o32(size[1]), o32(bbox[0]), o32(bbox[1]), o16(frame_duration), o16(1000), o8(frame_disposal), o8(frame_blend))
        seq_num += 1
        if frame == 0 and (not default_image):
            ImageFile._save(im_frame, _idat(fp, chunk), [('zip', (0, 0) + im_frame.size, 0, rawmode)])
        else:
            fdat_chunks = _fdat(fp, chunk, seq_num)
            ImageFile._save(im_frame, fdat_chunks, [('zip', (0, 0) + im_frame.size, 0, rawmode)])
            seq_num = fdat_chunks.seq_num