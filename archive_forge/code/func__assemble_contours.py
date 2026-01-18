import numpy as np
from ._find_contours_cy import _get_contour_segments
from collections import deque
def _assemble_contours(segments):
    current_index = 0
    contours = {}
    starts = {}
    ends = {}
    for from_point, to_point in segments:
        if from_point == to_point:
            continue
        tail, tail_num = starts.pop(to_point, (None, None))
        head, head_num = ends.pop(from_point, (None, None))
        if tail is not None and head is not None:
            if tail is head:
                head.append(to_point)
            elif tail_num > head_num:
                head.extend(tail)
                contours.pop(tail_num, None)
                starts[head[0]] = (head, head_num)
                ends[head[-1]] = (head, head_num)
            else:
                tail.extendleft(reversed(head))
                starts.pop(head[0], None)
                contours.pop(head_num, None)
                starts[tail[0]] = (tail, tail_num)
                ends[tail[-1]] = (tail, tail_num)
        elif tail is None and head is None:
            new_contour = deque((from_point, to_point))
            contours[current_index] = new_contour
            starts[from_point] = (new_contour, current_index)
            ends[to_point] = (new_contour, current_index)
            current_index += 1
        elif head is None:
            tail.appendleft(from_point)
            starts[from_point] = (tail, tail_num)
        else:
            head.append(to_point)
            ends[to_point] = (head, head_num)
    return [np.array(contour) for _, contour in sorted(contours.items())]