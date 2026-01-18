import struct
def _read_v3(handle):
    record = Record()
    record.version = 3
    section = ''
    for line in handle:
        line = line.rstrip('\r\n')
        if not line:
            continue
        if line.startswith('[HEADER]'):
            section = 'HEADER'
        elif line.startswith('[INTENSITY]'):
            section = 'INTENSITY'
            record.intensities = np.zeros((record.nrows, record.ncols))
            record.stdevs = np.zeros((record.nrows, record.ncols))
            record.npix = np.zeros((record.nrows, record.ncols), int)
        elif line.startswith('[MASKS]'):
            section = 'MASKS'
            record.mask = np.zeros((record.nrows, record.ncols), bool)
        elif line.startswith('[OUTLIERS]'):
            section = 'OUTLIERS'
            record.outliers = np.zeros((record.nrows, record.ncols), bool)
        elif line.startswith('[MODIFIED]'):
            section = 'MODIFIED'
            record.modified = np.zeros((record.nrows, record.ncols))
        elif line.startswith('['):
            raise ParserError('Unknown section found in version 3 CEL file')
        elif section == 'HEADER':
            key, value = line.split('=', 1)
            if key == 'Cols':
                record.ncols = int(value)
            elif key == 'Rows':
                record.nrows = int(value)
            elif key == 'GridCornerUL':
                x, y = value.split()
                record.GridCornerUL = (int(x), int(y))
            elif key == 'GridCornerUR':
                x, y = value.split()
                record.GridCornerUR = (int(x), int(y))
            elif key == 'GridCornerLR':
                x, y = value.split()
                record.GridCornerLR = (int(x), int(y))
            elif key == 'GridCornerLL':
                x, y = value.split()
                record.GridCornerLL = (int(x), int(y))
            elif key == 'DatHeader':
                record.DatHeader = {}
                i = value.find(':')
                if i >= 0:
                    min_max_pixel_intensity, filename = value[:i].split()
                    record.DatHeader['filename'] = filename
                    assert min_max_pixel_intensity[0] == '['
                    assert min_max_pixel_intensity[-1] == ']'
                    min_pixel_intensity, max_pixel_intensity = min_max_pixel_intensity[1:-1].split('..')
                    record.DatHeader['min-pixel_intensity'] = int(min_pixel_intensity)
                    record.DatHeader['max-pixel_intensity'] = int(max_pixel_intensity)
                    value = value[i + 1:]
                    index = 0
                    field = value[index:index + 9]
                    if field[:4] != 'CLS=' or field[8] != ' ':
                        raise ValueError("Field does not start with 'CLS=' or have a blank space at position 8")
                    record.DatHeader['CLS'] = int(field[4:8])
                    index += 9
                    field = value[index:index + 9]
                    if field[:4] != 'RWS=' or field[8] != ' ':
                        raise ValueError("Field does not start with 'RWS=' or have a blank space at position 8")
                    record.DatHeader['RWS'] = int(field[4:8])
                    index += 9
                    field = value[index:index + 7]
                    if field[:4] != 'XIN=' or field[6] != ' ':
                        raise ValueError("Field does not start with 'XIN=' or have a blank space at position 6")
                    record.DatHeader['XIN'] = int(field[4:6])
                    index += 7
                    field = value[index:index + 7]
                    if field[:4] != 'YIN=' or field[6] != ' ':
                        raise ValueError("Field does not start with 'YIN=' or have a blank space at poition 6")
                    record.DatHeader['YIN'] = int(field[4:6])
                    index += 7
                    field = value[index:index + 6]
                    if field[:3] != 'VE=' or field[5] != ' ':
                        raise ValueError("Field does not start with 'VE=' or have a blank space at position 5")
                    record.DatHeader['VE'] = int(field[3:5])
                    index += 6
                    field = value[index:index + 7]
                    if field[6] != ' ':
                        raise ValueError("Field value for position 6 isn't a blank space")
                    temperature = field[:6].strip()
                    if temperature:
                        record.DatHeader['temperature'] = int(temperature)
                    else:
                        record.DatHeader['temperature'] = None
                    index += 7
                    field = value[index:index + 4]
                    if not field.endswith(' '):
                        raise ValueError("Field doesn't end with a blank space")
                    record.DatHeader['laser-power'] = float(field)
                    index += 4
                    field = value[index:index + 18]
                    if field[8] != ' ':
                        raise ValueError("Field value for position 8 isn't a blank space")
                    record.DatHeader['scan-date'] = field[:8]
                    if field[17] != ' ':
                        raise ValueError("Field value for position 17 isn't a blank space")
                    record.DatHeader['scan-date'] = field[:8]
                    record.DatHeader['scan-time'] = field[9:17]
                    index += 18
                    value = value[index:]
                subfields = value.split('\x14')
                if len(subfields) != 12:
                    ValueError("Subfields length isn't 12")
                subfield = subfields[0]
                try:
                    scanner_id, scanner_type = subfield.split()
                except ValueError:
                    scanner_id = subfield.strip()
                else:
                    record.DatHeader['scanner-type'] = scanner_type
                record.DatHeader['scanner-id'] = scanner_id
                record.DatHeader['array-type'] = subfields[2].strip()
                field = subfields[7].strip()
                if field:
                    record.DatHeader['filter-wavelength'] = int(field)
                field = subfields[8].strip()
                if field:
                    record.DatHeader['arc-radius'] = float(field)
                field = subfields[9].strip()
                if field:
                    record.DatHeader['laser-spotsize'] = float(field)
                field = subfields[10].strip()
                if field:
                    record.DatHeader['pixel-size'] = float(field)
                field = subfields[11].strip()
                if field:
                    record.DatHeader['image-orientation'] = int(field)
            elif key == 'Algorithm':
                record.Algorithm = value
            elif key == 'AlgorithmParameters':
                parameters = value.split(';')
                values = {}
                for parameter in parameters:
                    key, value = parameter.split(':', 1)
                    if key in ('Percentile', 'CellMargin', 'FullFeatureWidth', 'FullFeatureHeight', 'PoolWidthExtenstion', 'PoolHeightExtension', 'NumPixelsToUse', 'ExtendPoolWidth', 'ExtendPoolHeight', 'OutlierRatioLowPercentile', 'OutlierRatioHighPercentile', 'HalfCellRowsDivisor', 'HalfCellRowsRemainder', 'HighCutoff', 'LowCutoff', 'featureRows', 'featureColumns'):
                        values[key] = int(value)
                    elif key in ('OutlierHigh', 'OutlierLow', 'StdMult', 'PercentileSpread', 'PairCutoff', 'featureWidth', 'featureHeight'):
                        values[key] = float(value)
                    elif key in ('FixedCellSize', 'IgnoreOutliersInShiftRows', 'FeatureExtraction', 'UseSubgrids', 'RandomizePixels', 'ImageCalibration', 'IgnoreShiftRowOutliers'):
                        if value == 'TRUE':
                            value = True
                        elif value == 'FALSE':
                            value = False
                        else:
                            raise ValueError('Unexpected boolean value')
                        values[key] = value
                    elif key in ('AlgVersion', 'ErrorBasis', 'CellIntensityCalculationType'):
                        values[key] = value
                    else:
                        raise ValueError('Unexpected tag in AlgorithmParameters')
                record.AlgorithmParameters = values
        elif section == 'INTENSITY':
            if line.startswith('NumberCells='):
                key, value = line.split('=', 1)
                record.NumberCells = int(value)
            elif line.startswith('CellHeader='):
                key, value = line.split('=', 1)
                if value.split() != ['X', 'Y', 'MEAN', 'STDV', 'NPIXELS']:
                    raise ParserError('Unexpected CellHeader in INTENSITY section CEL version 3 file')
            else:
                words = line.split()
                y = int(words[0])
                x = int(words[1])
                record.intensities[x, y] = float(words[2])
                record.stdevs[x, y] = float(words[3])
                record.npix[x, y] = int(words[4])
        elif section == 'MASKS':
            if line.startswith('NumberCells='):
                key, value = line.split('=', 1)
                record.nmask = int(value)
            elif line.startswith('CellHeader='):
                key, value = line.split('=', 1)
                if value.split() != ['X', 'Y']:
                    raise ParserError('Unexpected CellHeader in MASKS section in CEL version 3 file')
            else:
                words = line.split()
                y = int(words[0])
                x = int(words[1])
                record.mask[x, y] = True
        elif section == 'OUTLIERS':
            if line.startswith('NumberCells='):
                key, value = line.split('=', 1)
                record.noutliers = int(value)
            elif line.startswith('CellHeader='):
                key, value = line.split('=', 1)
                if value.split() != ['X', 'Y']:
                    raise ParserError('Unexpected CellHeader in OUTLIERS section in CEL version 3 file')
            else:
                words = line.split()
                y = int(words[0])
                x = int(words[1])
                record.outliers[x, y] = True
        elif section == 'MODIFIED':
            if line.startswith('NumberCells='):
                key, value = line.split('=', 1)
                record.nmodified = int(value)
            elif line.startswith('CellHeader='):
                key, value = line.split('=', 1)
                if value.split() != ['X', 'Y', 'ORIGMEAN']:
                    raise ParserError('Unexpected CellHeader in MODIFIED section in CEL version 3 file')
            else:
                words = line.split()
                y = int(words[0])
                x = int(words[1])
                record.modified[x, y] = float(words[2])
    return record