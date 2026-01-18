import sys
import time
import pygame as pg
def SpeedTest(image):
    print(f'\nImage Scaling Speed Test - Image Size {str(image.get_size())}\n')
    imgsize = [image.get_width(), image.get_height()]
    duration = 0.0
    for i in range(128):
        shrinkx = imgsize[0] * i // 128
        shrinky = imgsize[1] * i // 128
        start = time.time()
        tempimg = pg.transform.smoothscale(image, (shrinkx, shrinky))
        duration += time.time() - start
        del tempimg
    print(f'Average transform.smoothscale shrink time: {duration / 128 * 1000:.4f} ms.')
    duration = 0.0
    for i in range(128):
        expandx = imgsize[0] * (i + 129) // 128
        expandy = imgsize[1] * (i + 129) // 128
        start = time.time()
        tempimg = pg.transform.smoothscale(image, (expandx, expandy))
        duration += time.time() - start
        del tempimg
    print(f'Average transform.smoothscale expand time: {duration / 128 * 1000:.4f} ms.')
    duration = 0.0
    for i in range(128):
        shrinkx = imgsize[0] * i // 128
        shrinky = imgsize[1] * i // 128
        start = time.time()
        tempimg = pg.transform.scale(image, (shrinkx, shrinky))
        duration += time.time() - start
        del tempimg
    print(f'Average transform.scale shrink time: {duration / 128 * 1000:.4f} ms.')
    duration = 0.0
    for i in range(128):
        expandx = imgsize[0] * (i + 129) // 128
        expandy = imgsize[1] * (i + 129) // 128
        start = time.time()
        tempimg = pg.transform.scale(image, (expandx, expandy))
        duration += time.time() - start
        del tempimg
    print(f'Average transform.scale expand time: {duration / 128 * 1000:.4f} ms.')