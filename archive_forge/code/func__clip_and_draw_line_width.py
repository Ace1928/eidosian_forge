from collections import namedtuple
from math import floor, ceil
def _clip_and_draw_line_width(surf, rect, color, line, width):
    yinc = xinc = 0
    if abs(line[0] - line[2]) > abs(line[1] - line[3]):
        yinc = 1
    else:
        xinc = 1
    newpts = line[:]
    if _clip_and_draw_line(surf, rect, color, newpts):
        anydrawn = 1
        frame = newpts[:]
    else:
        anydrawn = 0
        frame = [10000, 10000, -10000, -10000]
    for loop in range(1, width // 2 + 1):
        newpts[0] = line[0] + xinc * loop
        newpts[1] = line[1] + yinc * loop
        newpts[2] = line[2] + xinc * loop
        newpts[3] = line[3] + yinc * loop
        if _clip_and_draw_line(surf, rect, color, newpts):
            anydrawn = 1
            frame[0] = min(newpts[0], frame[0])
            frame[1] = min(newpts[1], frame[1])
            frame[2] = max(newpts[2], frame[2])
            frame[3] = max(newpts[3], frame[3])
        if loop * 2 < width:
            newpts[0] = line[0] - xinc * loop
            newpts[1] = line[1] - yinc * loop
            newpts[2] = line[2] - xinc * loop
            newpts[3] = line[3] - yinc * loop
            if _clip_and_draw_line(surf, rect, color, newpts):
                anydrawn = 1
                frame[0] = min(newpts[0], frame[0])
                frame[1] = min(newpts[1], frame[1])
                frame[2] = max(newpts[2], frame[2])
                frame[3] = max(newpts[3], frame[3])
    return anydrawn