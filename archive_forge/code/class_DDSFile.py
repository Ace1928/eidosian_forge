from struct import pack, unpack, calcsize
class DDSFile(object):
    fields = (('size', 0), ('flags', 1), ('height', 2), ('width', 3), ('pitchOrLinearSize', 4), ('depth', 5), ('mipmapCount', 6), ('pf_size', 18), ('pf_flags', 19), ('pf_fourcc', 20), ('pf_rgbBitCount', 21), ('pf_rBitMask', 22), ('pf_gBitMask', 23), ('pf_bBitMask', 24), ('pf_aBitMask', 25), ('caps1', 26), ('caps2', 27))

    def __init__(self, filename=None):
        super(DDSFile, self).__init__()
        self._dxt = 0
        self._fmt = None
        self.meta = meta = QueryDict()
        self.count = 0
        self.images = []
        self.images_size = []
        for field, index in DDSFile.fields:
            meta[field] = 0
        if filename:
            self.load(filename)

    def load(self, filename):
        self.filename = filename
        with open(filename, 'rb') as fd:
            data = fd.read()
        if data[:4] != b'DDS ':
            raise DDSException('Invalid magic header {}'.format(data[:4]))
        fmt = 'I' * 31
        fmt_size = calcsize(fmt)
        pf_size = calcsize('I' * 8)
        header, data = (data[4:4 + fmt_size], data[4 + fmt_size:])
        if len(header) != fmt_size:
            raise DDSException('Truncated header in')
        header = unpack(fmt, header)
        meta = self.meta
        for name, index in DDSFile.fields:
            meta[name] = header[index]
        if meta.size != fmt_size:
            raise DDSException('Invalid header size (%d instead of %d)' % (meta.size, fmt_size))
        if meta.pf_size != pf_size:
            raise DDSException('Invalid pixelformat size (%d instead of %d)' % (meta.pf_size, pf_size))
        if not check_flags(meta.flags, DDSD_CAPS | DDSD_PIXELFORMAT | DDSD_WIDTH | DDSD_HEIGHT):
            raise DDSException('Not enough flags')
        if not check_flags(meta.caps1, DDSCAPS_TEXTURE):
            raise DDSException('Not a DDS texture')
        self.count = 1
        if check_flags(meta.flags, DDSD_MIPMAPCOUNT):
            if not check_flags(meta.caps1, DDSCAPS_COMPLEX | DDSCAPS_MIPMAP):
                raise DDSException('Invalid mipmap without flags')
            self.count = meta.mipmapCount
        hasrgb = check_flags(meta.pf_flags, DDPF_RGB)
        hasalpha = check_flags(meta.pf_flags, DDPF_ALPHAPIXELS)
        hasluminance = check_flags(meta.pf_flags, DDPF_LUMINANCE)
        bpp = None
        dxt = block = pitch = 0
        if hasrgb or hasalpha or hasluminance:
            bpp = meta.pf_rgbBitCount
        if hasrgb and hasluminance:
            raise DDSException('File have RGB and Luminance')
        if hasrgb:
            dxt = 0
        elif hasalpha and (not hasluminance):
            dxt = 1
        elif hasluminance and (not hasalpha):
            dxt = 2
        elif hasalpha and hasluminance:
            dxt = 3
        elif check_flags(meta.pf_flags, DDPF_FOURCC):
            dxt = meta.pf_fourcc
            if dxt not in (DDS_DXT1, DDS_DXT2, DDS_DXT3, DDS_DXT4, DDS_DXT5):
                raise DDSException('Unsupported FOURCC')
        else:
            raise DDSException('Unsupported format specified')
        if bpp:
            block = align_value(bpp, 8) // 8
            pitch = align_value(block * meta.width, 4)
        if check_flags(meta.flags, DDSD_LINEARSIZE):
            if dxt in (0, 1, 2, 3):
                size = pitch * meta.height
            else:
                size = dxt_size(meta.width, meta.height, dxt)
        w = meta.width
        h = meta.height
        images = self.images
        images_size = self.images_size
        for i in range(self.count):
            if dxt in (0, 1, 2, 3):
                size = align_value(block * w, 4) * h
            else:
                size = dxt_size(w, h, dxt)
            image, data = (data[:size], data[size:])
            if len(image) < size:
                raise DDSException('Truncated image for mipmap %d' % i)
            images_size.append((w, h))
            images.append(image)
            if w == 1 and h == 1:
                break
            w = max(1, w // 2)
            h = max(1, h // 2)
        if len(images) == 0:
            raise DDSException('No images available')
        if len(images) < self.count:
            raise DDSException('Not enough images')
        self._dxt = dxt

    def save(self, filename):
        if len(self.images) == 0:
            raise DDSException('No images to save')
        fields = dict(DDSFile.fields)
        fields_keys = list(fields.keys())
        fields_index = list(fields.values())
        mget = self.meta.get
        header = []
        for idx in range(31):
            if idx in fields_index:
                value = mget(fields_keys[fields_index.index(idx)], 0)
            else:
                value = 0
            header.append(value)
        with open(filename, 'wb') as fd:
            fd.write('DDS ')
            fd.write(pack('I' * 31, *header))
            for image in self.images:
                fd.write(image)

    def add_image(self, level, bpp, fmt, width, height, data):
        assert bpp == 32
        assert fmt in ('rgb', 'rgba', 'dxt1', 'dxt2', 'dxt3', 'dxt4', 'dxt5')
        assert width > 0
        assert height > 0
        assert level >= 0
        meta = self.meta
        images = self.images
        if len(images) == 0:
            assert level == 0
            for k in meta.keys():
                meta[k] = 0
            self._fmt = fmt
            meta.size = calcsize('I' * 31)
            meta.pf_size = calcsize('I' * 8)
            meta.pf_flags = 0
            meta.flags = DDSD_CAPS | DDSD_PIXELFORMAT | DDSD_WIDTH | DDSD_HEIGHT
            meta.width = width
            meta.height = height
            meta.caps1 = DDSCAPS_TEXTURE
            meta.flags |= DDSD_LINEARSIZE
            meta.pitchOrLinearSize = len(data)
            meta.pf_rgbBitCount = 32
            meta.pf_rBitMask = 16711680
            meta.pf_gBitMask = 65280
            meta.pf_bBitMask = 255
            meta.pf_aBitMask = 4278190080
            if fmt in ('rgb', 'rgba'):
                assert True
                assert bpp == 32
                meta.pf_flags |= DDPF_RGB
                meta.pf_rgbBitCount = 32
                meta.pf_rBitMask = 16711680
                meta.pf_gBitMask = 65280
                meta.pf_bBitMask = 255
                meta.pf_aBitMask = 0
                if fmt == 'rgba':
                    meta.pf_flags |= DDPF_ALPHAPIXELS
                    meta.pf_aBitMask = 4278190080
            else:
                meta.pf_flags |= DDPF_FOURCC
                if fmt == 'dxt1':
                    meta.pf_fourcc = DDS_DXT1
                elif fmt == 'dxt2':
                    meta.pf_fourcc = DDS_DXT2
                elif fmt == 'dxt3':
                    meta.pf_fourcc = DDS_DXT3
                elif fmt == 'dxt4':
                    meta.pf_fourcc = DDS_DXT4
                elif fmt == 'dxt5':
                    meta.pf_fourcc = DDS_DXT5
            images.append(data)
        else:
            assert level == len(images)
            assert fmt == self._fmt
            images.append(data)
            meta.flags |= DDSD_MIPMAPCOUNT
            meta.caps1 |= DDSCAPS_COMPLEX | DDSCAPS_MIPMAP
            meta.mipmapCount = len(images)

    def __repr__(self):
        return '<DDSFile filename=%r size=%r dxt=%r len(images)=%r>' % (self.filename, self.size, self.dxt, len(self.images))

    def _get_size(self):
        meta = self.meta
        return (meta.width, meta.height)

    def _set_size(self, size):
        self.meta.update({'width': size[0], 'height': size[1]})
    size = property(_get_size, _set_size)

    def _get_dxt(self):
        return dxt_to_str(self._dxt)

    def _set_dxt(self, dxt):
        self._dxt = str_to_dxt(dxt)
    dxt = property(_get_dxt, _set_dxt)