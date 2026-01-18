import os
import zlib
import time  # noqa
import logging
import numpy as np
class ShapeTag(DefinitionTag):

    def __init__(self, bitmapId, xy, wh):
        DefinitionTag.__init__(self)
        self.tagtype = 2
        self.bitmapId = bitmapId
        self.xy = xy
        self.wh = wh

    def process_tag(self):
        """Returns a defineshape tag. with a bitmap fill"""
        bb = bytes()
        bb += int2uint16(self.id)
        xy, wh = (self.xy, self.wh)
        tmp = self.make_rect_record(xy[0], wh[0], xy[1], wh[1])
        bb += tmp.tobytes()
        bb += int2uint8(1)
        bb += 'A'.encode('ascii')
        bb += int2uint16(self.bitmapId)
        bb += self.make_matrix_record(scale_xy=(20, 20)).tobytes()
        bb += int2uint8(0)
        bb += 'D'.encode('ascii')
        self.bytes = bb
        bits = BitArray()
        bits += self.make_style_change_record(0, 1, moveTo=(self.wh[0], self.wh[1]))
        bits += self.make_straight_edge_record(-self.wh[0], 0)
        bits += self.make_straight_edge_record(0, -self.wh[1])
        bits += self.make_straight_edge_record(self.wh[0], 0)
        bits += self.make_straight_edge_record(0, self.wh[1])
        bits += self.make_end_shape_record()
        self.bytes += bits.tobytes()

    def make_style_change_record(self, lineStyle=None, fillStyle=None, moveTo=None):
        bits = BitArray()
        bits += '0'
        bits += '0'
        if lineStyle:
            bits += '1'
        else:
            bits += '0'
        if fillStyle:
            bits += '1'
        else:
            bits += '0'
        bits += '0'
        if moveTo:
            bits += '1'
        else:
            bits += '0'
        if moveTo:
            bits += twits2bits([moveTo[0], moveTo[1]])
        if fillStyle:
            bits += int2bits(fillStyle, 4)
        if lineStyle:
            bits += int2bits(lineStyle, 4)
        return bits

    def make_straight_edge_record(self, *dxdy):
        if len(dxdy) == 1:
            dxdy = dxdy[0]
        xbits = signedint2bits(dxdy[0] * 20)
        ybits = signedint2bits(dxdy[1] * 20)
        nbits = max([len(xbits), len(ybits)])
        bits = BitArray()
        bits += '11'
        bits += int2bits(nbits - 2, 4)
        bits += '1'
        bits += signedint2bits(dxdy[0] * 20, nbits)
        bits += signedint2bits(dxdy[1] * 20, nbits)
        return bits

    def make_end_shape_record(self):
        bits = BitArray()
        bits += '0'
        bits += '0' * 5
        return bits