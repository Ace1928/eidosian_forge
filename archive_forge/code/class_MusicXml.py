from functools import reduce
from pyparsing import Word, OneOrMore, Optional, Literal, NotAny, MatchFirst
from pyparsing import Group, oneOf, Suppress, ZeroOrMore, Combine, FollowedBy
from pyparsing import srange, CharsNotIn, StringEnd, LineEnd, White, Regex
from pyparsing import nums, alphas, alphanums, ParseException, Forward
import types, sys, os, re, datetime
class MusicXml:
    typeMap = {1: 'long', 2: 'breve', 4: 'whole', 8: 'half', 16: 'quarter', 32: 'eighth', 64: '16th', 128: '32nd', 256: '64th'}
    dynaMap = {'p': 1, 'pp': 1, 'ppp': 1, 'pppp': 1, 'f': 1, 'ff': 1, 'fff': 1, 'ffff': 1, 'mp': 1, 'mf': 1, 'sfz': 1}
    tempoMap = {'larghissimo': 40, 'moderato': 104, 'adagissimo': 44, 'allegretto': 112, 'lentissimo': 48, 'allegro': 120, 'largo': 56, 'vivace': 168, 'adagio': 59, 'vivo': 180, 'lento': 62, 'presto': 192, 'larghetto': 66, 'allegrissimo': 208, 'adagietto': 76, 'vivacissimo': 220, 'andante': 88, 'prestissimo': 240, 'andantino': 96}
    wedgeMap = {'>(': 1, '>)': 1, '<(': 1, '<)': 1, 'crescendo(': 1, 'crescendo)': 1, 'diminuendo(': 1, 'diminuendo)': 1}
    artMap = {'.': 'staccato', '>': 'accent', 'accent': 'accent', 'wedge': 'staccatissimo', 'tenuto': 'tenuto', 'breath': 'breath-mark', 'marcato': 'strong-accent', '^': 'strong-accent', 'slide': 'scoop'}
    ornMap = {'trill': 'trill-mark', 'T': 'trill-mark', 'turn': 'turn', 'uppermordent': 'inverted-mordent', 'lowermordent': 'mordent', 'pralltriller': 'inverted-mordent', 'mordent': 'mordent', 'turn': 'turn', 'invertedturn': 'inverted-turn'}
    tecMap = {'upbow': 'up-bow', 'downbow': 'down-bow', 'plus': 'stopped', 'open': 'open-string', 'snap': 'snap-pizzicato', 'thumb': 'thumb-position'}
    capoMap = {'fine': ('Fine', 'fine', 'yes'), 'D.S.': ('D.S.', 'dalsegno', 'segno'), 'D.C.': ('D.C.', 'dacapo', 'yes'), 'dacapo': ('D.C.', 'dacapo', 'yes'), 'dacoda': ('To Coda', 'tocoda', 'coda'), 'coda': ('coda', 'coda', 'coda'), 'segno': ('segno', 'segno', 'segno')}
    sharpness = ['Fb', 'Cb', 'Gb', 'Db', 'Ab', 'Eb', 'Bb', 'F', 'C', 'G', 'D', 'A', 'E', 'B', 'F#', 'C#', 'G#', 'D#', 'A#', 'E#', 'B#']
    offTab = {'maj': 8, 'm': 11, 'min': 11, 'mix': 9, 'dor': 10, 'phr': 12, 'lyd': 7, 'loc': 13}
    modTab = {'maj': 'major', 'm': 'minor', 'min': 'minor', 'mix': 'mixolydian', 'dor': 'dorian', 'phr': 'phrygian', 'lyd': 'lydian', 'loc': 'locrian'}
    clefMap = {'alto1': ('C', '1'), 'alto2': ('C', '2'), 'alto': ('C', '3'), 'alto4': ('C', '4'), 'tenor': ('C', '4'), 'bass3': ('F', '3'), 'bass': ('F', '4'), 'treble': ('G', '2'), 'perc': ('percussion', ''), 'none': ('', ''), 'tab': ('TAB', '5')}
    clefLineMap = {'B': 'treble', 'G': 'alto1', 'E': 'alto2', 'C': 'alto', 'A': 'tenor', 'F': 'bass3', 'D': 'bass'}
    alterTab = {'=': '0', '_': '-1', '__': '-2', '^': '1', '^^': '2'}
    accTab = {'=': 'natural', '_': 'flat', '__': 'flat-flat', '^': 'sharp', '^^': 'sharp-sharp'}
    chordTab = compChordTab()
    uSyms = {'~': 'roll', 'H': 'fermata', 'L': '>', 'M': 'lowermordent', 'O': 'coda', 'P': 'uppermordent', 'S': 'segno', 'T': 'trill', 'u': 'upbow', 'v': 'downbow'}
    pageFmtDef = [0.75, 297, 210, 18, 18, 10, 10]
    metaTab = {'O': 'origin', 'A': 'area', 'Z': 'transcription', 'N': 'notes', 'G': 'group', 'H': 'history', 'R': 'rhythm', 'B': 'book', 'D': 'discography', 'F': 'fileurl', 'S': 'source', 'P': 'partmap', 'W': 'lyrics'}
    metaMap = {'C': 'composer'}
    metaTypes = {'composer': 1, 'lyricist': 1, 'poet': 1, 'arranger': 1, 'translator': 1, 'rights': 1}
    tuningDef = 'E2,A2,D3,G3,B3,E4'.split(',')

    def __init__(s):
        s.pageFmtCmd = []
        s.reset()

    def reset(s, fOpt=False):
        s.divisions = 2520
        s.ties = {}
        s.slurstack = {}
        s.slurbeg = []
        s.tmnum = 0
        s.tmden = 0
        s.ntup = 0
        s.trem = 0
        s.intrem = 0
        s.tupnts = []
        s.irrtup = 0
        s.ntype = ''
        s.unitL = (1, 8)
        s.unitLcur = (1, 8)
        s.keyAlts = {}
        s.msreAlts = {}
        s.curVolta = ''
        s.title = ''
        s.creator = {}
        s.metadata = {}
        s.lyrdash = {}
        s.usrSyms = s.uSyms
        s.prevNote = None
        s.prevLyric = {}
        s.grcbbrk = False
        s.linebrk = 0
        s.nextdecos = []
        s.prevmsre = None
        s.supports_tag = 0
        s.staveDefs = []
        s.staves = []
        s.groups = []
        s.grands = []
        s.gStaffNums = {}
        s.gNstaves = {}
        s.pageFmtAbc = []
        s.mdur = (4, 4)
        s.gtrans = 0
        s.midprg = ['', '', '', '']
        s.vid = ''
        s.pid = ''
        s.gcue_on = 0
        s.percVoice = 0
        s.percMap = {}
        s.pMapFound = 0
        s.vcepid = {}
        s.midiInst = {}
        s.capo = 0
        s.tunmid = []
        s.tunTup = []
        s.fOpt = fOpt
        s.orderChords = 0
        s.chordDecos = {}
        ch10 = 'acoustic-bass-drum,35;bass-drum-1,36;side-stick,37;acoustic-snare,38;hand-clap,39;electric-snare,40;low-floor-tom,41;closed-hi-hat,42;high-floor-tom,43;pedal-hi-hat,44;low-tom,45;open-hi-hat,46;low-mid-tom,47;hi-mid-tom,48;crash-cymbal-1,49;high-tom,50;ride-cymbal-1,51;chinese-cymbal,52;ride-bell,53;tambourine,54;splash-cymbal,55;cowbell,56;crash-cymbal-2,57;vibraslap,58;ride-cymbal-2,59;hi-bongo,60;low-bongo,61;mute-hi-conga,62;open-hi-conga,63;low-conga,64;high-timbale,65;low-timbale,66;high-agogo,67;low-agogo,68;cabasa,69;maracas,70;short-whistle,71;long-whistle,72;short-guiro,73;long-guiro,74;claves,75;hi-wood-block,76;low-wood-block,77;mute-cuica,78;open-cuica,79;mute-triangle,80;open-triangle,81'
        s.percsnd = [x.split(',') for x in ch10.split(';')]
        s.gTime = (0, 0)
        s.tabStaff = ''

    def mkPitch(s, acc, note, oct, lev):
        if s.percVoice:
            octq = int(oct) + s.gtrans
            tup = s.percMap.get((s.pid, acc + note, octq), s.percMap.get(('', acc + note, octq), 0))
            if tup:
                step, soct, midi, notehead = tup
            else:
                step, soct = (note, octq)
            octnum = (4 if step.upper() == step else 5) + int(soct)
            if not tup:
                midi = str(octnum * 12 + [0, 2, 4, 5, 7, 9, 11]['CDEFGAB'.index(step.upper())] + {'^': 1, '_': -1}.get(acc, 0) + 12)
                notehead = {'^': 'x', '_': 'circle-x'}.get(acc, 'normal')
                if s.pMapFound:
                    info('no I:percmap for: %s%s in part %s, voice %s' % (acc + note, -oct * ',' if oct < 0 else oct * "'", s.pid, s.vid))
                s.percMap[s.pid, acc + note, octq] = (note, octq, midi, notehead)
            else:
                step, octnum = stepTrans(step.upper(), octnum, s.curClef)
            pitch = E.Element('unpitched')
            addElemT(pitch, 'display-step', step.upper(), lev + 1)
            addElemT(pitch, 'display-octave', str(octnum), lev + 1)
            return (pitch, '', midi, notehead)
        nUp = note.upper()
        octnum = (4 if nUp == note else 5) + int(oct) + s.gtrans
        pitch = E.Element('pitch')
        addElemT(pitch, 'step', nUp, lev + 1)
        alter = ''
        if (note, oct) in s.ties:
            tied_alter, _, vnum, _ = s.ties[note, oct]
            if vnum == s.overlayVnum:
                alter = tied_alter
        elif acc:
            s.msreAlts[nUp, octnum] = s.alterTab[acc]
            alter = s.alterTab[acc]
        elif (nUp, octnum) in s.msreAlts:
            alter = s.msreAlts[nUp, octnum]
        elif nUp in s.keyAlts:
            alter = s.keyAlts[nUp]
        if alter:
            addElemT(pitch, 'alter', alter, lev + 1)
        addElemT(pitch, 'octave', str(octnum), lev + 1)
        return (pitch, alter, '', '')

    def getNoteDecos(s, n):
        decos = s.nextdecos
        ndeco = getattr(n, 'deco', 0)
        if ndeco:
            decos += [s.usrSyms.get(d, d).strip('!+') for d in ndeco.t]
        s.nextdecos = []
        if s.tabStaff == s.pid and s.fOpt and (n.name != 'rest'):
            if [d for d in decos if d in '0123456789'] == []:
                decos.append('0')
        return decos

    def mkNote(s, n, lev):
        isgrace = getattr(n, 'grace', '')
        ischord = getattr(n, 'chord', '')
        if s.ntup >= 0 and (not isgrace) and (not ischord):
            s.ntup -= 1
            if s.ntup == -1 and s.trem <= 0:
                s.intrem = 0
        nnum, nden = n.dur.t
        if s.intrem:
            nnum += nnum
        if nden == 0:
            nden = 1
        num, den = simplify(nnum * s.unitLcur[0], nden * s.unitLcur[1])
        if den > 64:
            num = int(round(64 * float(num) / den))
            num, den = simplify(max([num, 1]), 64)
            info('duration too small: rounded to %d/%d' % (num, den))
        if n.name == 'rest' and ('Z' in n.t or 'X' in n.t):
            num, den = s.mdur
        noMsrRest = not (n.name == 'rest' and (num, den) == s.mdur)
        dvs = 4 * s.divisions * num // den
        rdvs = dvs
        num, den = simplify(num, den * 4)
        ndot = 0
        if num == 3 and noMsrRest:
            ndot = 1
            den = den // 2
        if num == 7 and noMsrRest:
            ndot = 2
            den = den // 4
        nt = E.Element('note')
        if isgrace:
            grace = E.Element('grace')
            if s.acciatura:
                grace.set('slash', 'yes')
                s.acciatura = 0
            addElem(nt, grace, lev + 1)
            dvs = rdvs = 0
            if den <= 16:
                den = 32
        if s.gcue_on:
            cue = E.Element('cue')
            addElem(nt, cue, lev + 1)
        if ischord:
            chord = E.Element('chord')
            addElem(nt, chord, lev + 1)
            rdvs = 0
        if den not in s.typeMap:
            info('illegal duration %d/%d' % (nnum, nden))
            den = min((x for x in s.typeMap.keys() if x > den))
        xmltype = str(s.typeMap[den])
        acc, step, oct = ('', 'C', '0')
        alter, midi, notehead = ('', '', '')
        if n.name == 'rest':
            if 'x' in n.t or 'X' in n.t:
                nt.set('print-object', 'no')
            rest = E.Element('rest')
            if not noMsrRest:
                rest.set('measure', 'yes')
            addElem(nt, rest, lev + 1)
        else:
            p = n.pitch.t
            if len(p) == 3:
                acc, step, oct = p
            else:
                step, oct = p
            pitch, alter, midi, notehead = s.mkPitch(acc, step, oct, lev + 1)
            if midi:
                acc = ''
            addElem(nt, pitch, lev + 1)
        if s.ntup >= 0:
            dvs = dvs * s.tmden // s.tmnum
        if dvs:
            addElemT(nt, 'duration', str(dvs), lev + 1)
            if not ischord:
                s.gTime = (s.gTime[1], s.gTime[1] + dvs)
        ptup = (step, oct)
        tstop = ptup in s.ties and s.ties[ptup][2] == s.overlayVnum
        if tstop:
            tie = E.Element('tie', type='stop')
            addElem(nt, tie, lev + 1)
        if getattr(n, 'tie', 0):
            tie = E.Element('tie', type='start')
            addElem(nt, tie, lev + 1)
        if (s.midprg != ['', '', '', ''] or midi) and n.name != 'rest':
            instId = 'I%s-%s' % (s.pid, 'X' + midi if midi else s.vid)
            chan, midi = ('10', midi) if midi else s.midprg[:2]
            inst = E.Element('instrument', id=instId)
            addElem(nt, inst, lev + 1)
            if instId not in s.midiInst:
                s.midiInst[instId] = (s.pid, s.vid, chan, midi, s.midprg[2], s.midprg[3])
        addElemT(nt, 'voice', '1', lev + 1)
        if noMsrRest:
            addElemT(nt, 'type', xmltype, lev + 1)
        for i in range(ndot):
            dot = E.Element('dot')
            addElem(nt, dot, lev + 1)
        decos = s.getNoteDecos(n)
        if acc and (not tstop):
            e = E.Element('accidental')
            if 'courtesy' in decos:
                e.set('parentheses', 'yes')
                decos.remove('courtesy')
            e.text = s.accTab[acc]
            addElem(nt, e, lev + 1)
        tupnotation = ''
        if s.ntup >= 0:
            tmod = mkTmod(s.tmnum, s.tmden, lev + 1)
            addElem(nt, tmod, lev + 1)
            if s.ntup > 0 and (not s.tupnts):
                tupnotation = 'start'
            s.tupnts.append((rdvs, tmod))
            if s.ntup == 0:
                if rdvs:
                    tupnotation = 'stop'
                s.cmpNormType(rdvs, lev + 1)
        hasStem = 1
        if not ischord:
            s.chordDecos = {}
        if 'stemless' in decos or (s.nostems and n.name != 'rest') or 'stemless' in s.chordDecos:
            hasStem = 0
            addElemT(nt, 'stem', 'none', lev + 1)
            if 'stemless' in decos:
                decos.remove('stemless')
            if hasattr(n, 'pitches'):
                s.chordDecos['stemless'] = 1
        if notehead:
            nh = addElemT(nt, 'notehead', re.sub('[+-]$', '', notehead), lev + 1)
            if notehead[-1] in '+-':
                nh.set('filled', 'yes' if notehead[-1] == '+' else 'no')
        gstaff = s.gStaffNums.get(s.vid, 0)
        if gstaff:
            addElemT(nt, 'staff', str(gstaff), lev + 1)
        if hasStem:
            s.doBeams(n, nt, den, lev + 1)
        s.doNotations(n, decos, ptup, alter, tupnotation, tstop, nt, lev + 1)
        if n.objs:
            s.doLyr(n, nt, lev + 1)
        else:
            s.prevLyric = {}
        return nt

    def cmpNormType(s, rdvs, lev):
        if rdvs:
            durs = [dur for dur, tmod in s.tupnts if dur > 0]
            ndur = sum(durs) // s.tmnum
            s.irrtup = any((dur != ndur for dur in durs))
            tix = 16 * s.divisions // ndur
            if tix in s.typeMap:
                s.ntype = str(s.typeMap[tix])
            else:
                s.irrtup = 0
        if s.irrtup:
            for dur, tmod in s.tupnts:
                addElemT(tmod, 'normal-type', s.ntype, lev + 1)
        s.tupnts = []

    def doNotations(s, n, decos, ptup, alter, tupnotation, tstop, nt, lev):
        slurs = getattr(n, 'slurs', 0)
        pts = getattr(n, 'pitches', [])
        ov = s.overlayVnum
        if pts:
            if type(pts.pitch) == pObj:
                pts = [pts.pitch]
            else:
                pts = [tuple(p.t[-2:]) for p in pts.pitch]
        for pt, (tie_alter, nts, vnum, ntelm) in sorted(list(s.ties.items())):
            if vnum != s.overlayVnum:
                continue
            if pts and pt in pts:
                continue
            if getattr(n, 'chord', 0):
                continue
            if pt == ptup:
                continue
            if getattr(n, 'grace', 0):
                continue
            info('tie between different pitches: %s%s converted to slur' % pt)
            del s.ties[pt]
            e = [t for t in ntelm.findall('tie') if t.get('type') == 'start'][0]
            ntelm.remove(e)
            e = [t for t in nts.findall('tied') if t.get('type') == 'start'][0]
            e.tag = 'slur'
            slurnum = pushSlur(s.slurstack, ov)
            e.set('number', str(slurnum))
            if slurs:
                slurs.t.append(')')
            else:
                slurs = pObj('slurs', [')'])
        tstart = getattr(n, 'tie', 0)
        if not (tstop or tstart or decos or slurs or s.slurbeg or tupnotation or s.trem):
            return nt
        nots = E.Element('notations')
        if s.trem:
            if s.trem < 0:
                tupnotation = 'single'
                s.trem = -s.trem
            if not tupnotation:
                return
            orn = E.Element('ornaments')
            trm = E.Element('tremolo', type=tupnotation)
            trm.text = str(s.trem)
            addElem(nots, orn, lev + 1)
            addElem(orn, trm, lev + 2)
            if tupnotation == 'stop' or tupnotation == 'single':
                s.trem = 0
        elif tupnotation:
            tup = E.Element('tuplet', type=tupnotation)
            if tupnotation == 'start':
                tup.set('bracket', 'yes')
            addElem(nots, tup, lev + 1)
        if tstop:
            del s.ties[ptup]
            tie = E.Element('tied', type='stop')
            addElem(nots, tie, lev + 1)
        if tstart:
            s.ties[ptup] = (alter, nots, s.overlayVnum, nt)
            tie = E.Element('tied', type='start')
            if tstart.t[0] == '.-':
                tie.set('line-type', 'dotted')
            addElem(nots, tie, lev + 1)
        if decos:
            slurMap = {'(': 1, '.(': 1, '(,': 1, "('": 1, '.(,': 1, ".('": 1}
            arts = []
            for d in decos:
                if d in slurMap:
                    s.slurbeg.append(d)
                    continue
                elif d == 'fermata' or d == 'H':
                    ntn = E.Element('fermata', type='upright')
                elif d == 'arpeggio':
                    ntn = E.Element('arpeggiate', number='1')
                elif d in ['~(', '~)']:
                    if d[1] == '(':
                        tp = 'start'
                        s.glisnum += 1
                        gn = s.glisnum
                    else:
                        tp = 'stop'
                        gn = s.glisnum
                        s.glisnum -= 1
                    if s.glisnum < 0:
                        s.glisnum = 0
                        continue
                    ntn = E.Element('glissando', {'line-type': 'wavy', 'number': '%d' % gn, 'type': tp})
                elif d in ['-(', '-)']:
                    if d[1] == '(':
                        tp = 'start'
                        s.slidenum += 1
                        gn = s.slidenum
                    else:
                        tp = 'stop'
                        gn = s.slidenum
                        s.slidenum -= 1
                    if s.slidenum < 0:
                        s.slidenum = 0
                        continue
                    ntn = E.Element('slide', {'line-type': 'solid', 'number': '%d' % gn, 'type': tp})
                else:
                    arts.append(d)
                    continue
                addElem(nots, ntn, lev + 1)
            if arts:
                rest = s.doArticulations(nt, nots, arts, lev + 1)
                if rest:
                    info('unhandled note decorations: %s' % rest)
        if slurs:
            for d in slurs.t:
                if not s.slurstack.get(ov, 0):
                    break
                slurnum = s.slurstack[ov].pop()
                slur = E.Element('slur', number='%d' % slurnum, type='stop')
                addElem(nots, slur, lev + 1)
        while s.slurbeg:
            stp = s.slurbeg.pop(0)
            slurnum = pushSlur(s.slurstack, ov)
            ntn = E.Element('slur', number='%d' % slurnum, type='start')
            if '.' in stp:
                ntn.set('line-type', 'dotted')
            if ',' in stp:
                ntn.set('placement', 'below')
            if "'" in stp:
                ntn.set('placement', 'above')
            addElem(nots, ntn, lev + 1)
        if list(nots) != []:
            addElem(nt, nots, lev)

    def doArticulations(s, nt, nots, arts, lev):
        decos = []
        for a in arts:
            if a in s.artMap:
                art = E.Element('articulations')
                addElem(nots, art, lev)
                addElem(art, E.Element(s.artMap[a]), lev + 1)
            elif a in s.ornMap:
                orn = E.Element('ornaments')
                addElem(nots, orn, lev)
                addElem(orn, E.Element(s.ornMap[a]), lev + 1)
            elif a in ['trill(', 'trill)']:
                orn = E.Element('ornaments')
                addElem(nots, orn, lev)
                type = 'start' if a.endswith('(') else 'stop'
                if type == 'start':
                    addElem(orn, E.Element('trill-mark'), lev + 1)
                addElem(orn, E.Element('wavy-line', type=type), lev + 1)
            elif a in s.tecMap:
                tec = E.Element('technical')
                addElem(nots, tec, lev)
                addElem(tec, E.Element(s.tecMap[a]), lev + 1)
            elif a in '0123456':
                tec = E.Element('technical')
                addElem(nots, tec, lev)
                if s.tabStaff == s.pid:
                    alt = int(nt.findtext('pitch/alter') or 0)
                    step = nt.findtext('pitch/step')
                    oct = int(nt.findtext('pitch/octave'))
                    midi = oct * 12 + [0, 2, 4, 5, 7, 9, 11]['CDEFGAB'.index(step)] + alt + 12
                    if a == '0':
                        firstFit = ''
                        for smid, istr in s.tunTup:
                            if midi >= smid:
                                isvrij = s.strAlloc.isVrij(istr - 1, s.gTime[0], s.gTime[1])
                                a = str(istr)
                                if not firstFit:
                                    firstFit = a
                                if isvrij:
                                    break
                        if not isvrij:
                            a = firstFit
                            s.strAlloc.bezet(int(a) - 1, s.gTime[0], s.gTime[1])
                    else:
                        s.strAlloc.bezet(int(a) - 1, s.gTime[0], s.gTime[1])
                    bmidi = s.tunmid[int(a) - 1]
                    fret = midi - bmidi
                    if fret < 25 and fret >= 0:
                        addElemT(tec, 'fret', str(fret), lev + 1)
                    else:
                        altp = 'b' if alt == -1 else '#' if alt == 1 else ''
                        info('fret %d out of range, note %s%d on string %s' % (fret, step + altp, oct, a))
                    addElemT(tec, 'string', a, lev + 1)
                else:
                    addElemT(tec, 'fingering', a, lev + 1)
            else:
                decos.append(a)
        return decos

    def doLyr(s, n, nt, lev):
        for i, lyrobj in enumerate(n.objs):
            lyrel = E.Element('lyric', number=str(i + 1))
            if lyrobj.name == 'syl':
                dash = len(lyrobj.t) == 2
                if dash:
                    if i in s.lyrdash:
                        type = 'middle'
                    else:
                        type = 'begin'
                        s.lyrdash[i] = 1
                elif i in s.lyrdash:
                    type = 'end'
                    del s.lyrdash[i]
                else:
                    type = 'single'
                addElemT(lyrel, 'syllabic', type, lev + 1)
                txt = lyrobj.t[0]
                txt = re.sub('(?<!\\\\)~', ' ', txt)
                txt = re.sub('\\\\(.)', '\\1', txt)
                addElemT(lyrel, 'text', txt, lev + 1)
            elif lyrobj.name == 'ext' and i in s.prevLyric:
                pext = s.prevLyric[i].find('extend')
                if pext == None:
                    ext = E.Element('extend', type='start')
                    addElem(s.prevLyric[i], ext, lev + 1)
                elif pext.get('type') == 'stop':
                    pext.set('type', 'continue')
                ext = E.Element('extend', type='stop')
                addElem(lyrel, ext, lev + 1)
            elif lyrobj.name == 'ext':
                info('lyric extend error')
                continue
            else:
                continue
            addElem(nt, lyrel, lev)
            s.prevLyric[i] = lyrel

    def doBeams(s, n, nt, den, lev):
        if hasattr(n, 'chord') or hasattr(n, 'grace'):
            s.grcbbrk = s.grcbbrk or n.bbrk.t[0]
            return
        bbrk = s.grcbbrk or n.bbrk.t[0] or den < 32
        s.grcbbrk = False
        if not s.prevNote:
            pbm = None
        else:
            pbm = s.prevNote.find('beam')
        bm = E.Element('beam', number='1')
        bm.text = 'begin'
        if pbm != None:
            if bbrk:
                if pbm.text == 'begin':
                    s.prevNote.remove(pbm)
                elif pbm.text == 'continue':
                    pbm.text = 'end'
                s.prevNote = None
            else:
                bm.text = 'continue'
        if den >= 32 and n.name != 'rest':
            addElem(nt, bm, lev)
            s.prevNote = nt

    def stopBeams(s):
        if not s.prevNote:
            return
        pbm = s.prevNote.find('beam')
        if pbm != None:
            if pbm.text == 'begin':
                s.prevNote.remove(pbm)
            elif pbm.text == 'continue':
                pbm.text = 'end'
        s.prevNote = None

    def staffDecos(s, decos, maat, lev):
        gstaff = s.gStaffNums.get(s.vid, 0)
        for d in decos:
            d = s.usrSyms.get(d, d).strip('!+')
            if d in s.dynaMap:
                dynel = E.Element('dynamics')
                addDirection(maat, dynel, lev, gstaff, [E.Element(d)], 'below', s.gcue_on)
            elif d in s.wedgeMap:
                if ')' in d:
                    type = 'stop'
                else:
                    type = 'crescendo' if '<' in d or 'crescendo' in d else 'diminuendo'
                addDirection(maat, E.Element('wedge', type=type), lev, gstaff)
            elif d.startswith('8v'):
                if 'a' in d:
                    type, plce = ('down', 'above')
                else:
                    type, plce = ('up', 'below')
                if ')' in d:
                    type = 'stop'
                addDirection(maat, E.Element('octave-shift', type=type, size='8'), lev, gstaff, placement=plce)
            elif d in ['ped', 'ped-up']:
                type = 'stop' if d.endswith('up') else 'start'
                addDirection(maat, E.Element('pedal', type=type), lev, gstaff)
            elif d in ['coda', 'segno']:
                text, attr, val = s.capoMap[d]
                dir = addDirection(maat, E.Element(text), lev, gstaff, placement='above')
                sound = E.Element('sound')
                sound.set(attr, val)
                addElem(dir, sound, lev + 1)
            elif d in s.capoMap:
                text, attr, val = s.capoMap[d]
                words = E.Element('words')
                words.text = text
                dir = addDirection(maat, words, lev, gstaff, placement='above')
                sound = E.Element('sound')
                sound.set(attr, val)
                addElem(dir, sound, lev + 1)
            elif d == '(' or d == '.(':
                s.slurbeg.append(d)
            elif d in ['/-', '//-', '///-', '////-']:
                s.tmnum, s.tmden, s.ntup, s.trem, s.intrem = (2, 1, 2, len(d) - 1, 1)
            elif d in ['/', '//', '///']:
                s.trem = -len(d)
            else:
                s.nextdecos.append(d)

    def doFields(s, maat, fieldmap, lev):

        def instDir(midelm, midnum, dirtxt):
            instId = 'I%s-%s' % (s.pid, s.vid)
            words = E.Element('words')
            words.text = dirtxt % midnum
            snd = E.Element('sound')
            mi = E.Element('midi-instrument', id=instId)
            dir = addDirection(maat, words, lev, gstaff, placement='above')
            addElem(dir, snd, lev + 1)
            addElem(snd, mi, lev + 2)
            addElemT(mi, midelm, midnum, lev + 3)

        def addTrans(n):
            e = E.Element('transpose')
            addElemT(e, 'chromatic', n, lev + 2)
            atts.append((9, e))

        def doClef(field):
            if re.search('perc|map', field):
                r = re.search('(perc|map)\\s*=\\s*(\\S*)', field)
                s.percVoice = 0 if r and r.group(2) not in ['on', 'true', 'perc'] else 1
                field = re.sub('(perc|map)\\s*=\\s*(\\S*)', '', field)
            clef, gtrans = (0, 0)
            clefn = re.search('alto1|alto2|alto4|alto|tenor|bass3|bass|treble|perc|none|tab', field)
            clefm = re.search("(?:^m=| m=|middle=)([A-Ga-g])([,']*)", field)
            trans_oct2 = re.search('octave=([-+]?\\d)', field)
            trans = re.search('(?:^t=| t=|transpose=)(-?[\\d]+)', field)
            trans_oct = re.search('([+-^_])(8|15)', field)
            cue_onoff = re.search('cue=(on|off)', field)
            strings = re.search('strings=(\\S+)', field)
            stafflines = re.search('stafflines=\\s*(\\d)', field)
            capo = re.search('capo=(\\d+)', field)
            if clefn:
                clef = clefn.group()
            if clefm:
                note, octstr = clefm.groups()
                nUp = note.upper()
                octnum = (4 if nUp == note else 5) + (len(octstr) if "'" in octstr else -len(octstr))
                gtrans = (3 if nUp in 'AFD' else 4) - octnum
                if clef not in ['perc', 'none']:
                    clef = s.clefLineMap[nUp]
            if clef:
                s.gtrans = gtrans
                if clef != 'none':
                    s.curClef = clef
                sign, line = s.clefMap[clef]
                if not sign:
                    return
                c = E.Element('clef')
                if gstaff:
                    c.set('number', str(gstaff))
                addElemT(c, 'sign', sign, lev + 2)
                if line:
                    addElemT(c, 'line', line, lev + 2)
                if trans_oct:
                    n = trans_oct.group(1) in '-_' and -1 or 1
                    if trans_oct.group(2) == '15':
                        n *= 2
                    addElemT(c, 'clef-octave-change', str(n), lev + 2)
                    if trans_oct.group(1) in '+-':
                        s.gtrans += n
                atts.append((7, c))
            if trans_oct2:
                n = int(trans_oct2.group(1))
                s.gtrans = gtrans + n
            if trans != None:
                e = E.Element('transpose')
                addElemT(e, 'chromatic', str(trans.group(1)), lev + 3)
                atts.append((9, e))
            if cue_onoff:
                s.gcue_on = cue_onoff.group(1) == 'on'
            nlines = 0
            if clef == 'tab':
                s.tabStaff = s.pid
                if capo:
                    s.capo = int(capo.group(1))
                if strings:
                    s.tuning = strings.group(1).split(',')
                s.tunmid = [int(boct) * 12 + [0, 2, 4, 5, 7, 9, 11]['CDEFGAB'.index(bstep)] + 12 + s.capo for bstep, boct in s.tuning]
                s.tunTup = sorted(zip(s.tunmid, range(len(s.tunmid), 0, -1)), reverse=1)
                s.tunmid.reverse()
                nlines = str(len(s.tuning))
                s.strAlloc.setlines(len(s.tuning), s.pid)
                s.nostems = 'nostems' in field
                s.diafret = 'diafret' in field
            if stafflines or nlines:
                e = E.Element('staff-details')
                if gstaff:
                    e.set('number', str(gstaff))
                if not nlines:
                    nlines = stafflines.group(1)
                addElemT(e, 'staff-lines', nlines, lev + 2)
                if clef == 'tab':
                    for line, t in enumerate(s.tuning):
                        st = E.Element('staff-tuning', line=str(line + 1))
                        addElemT(st, 'tuning-step', t[0], lev + 3)
                        addElemT(st, 'tuning-octave', t[1], lev + 3)
                        addElem(e, st, lev + 2)
                if s.capo:
                    addElemT(e, 'capo', str(s.capo), lev + 2)
                atts.append((8, e))
        s.diafret = 0
        atts = []
        gstaff = s.gStaffNums.get(s.vid, 0)
        for ftype, field in fieldmap.items():
            if not field:
                continue
            if ftype == 'Div':
                d = E.Element('divisions')
                d.text = field
                atts.append((1, d))
            elif ftype == 'gstaff':
                e = E.Element('staves')
                e.text = str(field)
                atts.append((4, e))
            elif ftype == 'M':
                if field == 'none':
                    continue
                if field == 'C':
                    field = '4/4'
                elif field == 'C|':
                    field = '2/2'
                t = E.Element('time')
                if '/' not in field:
                    info('M:%s not recognized, 4/4 assumed' % field)
                    field = '4/4'
                beats, btype = field.split('/')[:2]
                try:
                    s.mdur = simplify(eval(beats), int(btype))
                except:
                    info('error in M:%s, 4/4 assumed' % field)
                    s.mdur = (4, 4)
                    beats, btype = ('4', '4')
                addElemT(t, 'beats', beats, lev + 2)
                addElemT(t, 'beat-type', btype, lev + 2)
                atts.append((3, t))
            elif ftype == 'K':
                accs = ['F', 'C', 'G', 'D', 'A', 'E', 'B']
                mode = ''
                key = re.match('\\s*([A-G][#b]?)\\s*([a-zA-Z]*)', field)
                alts = re.search('\\s((\\s?[=^_][A-Ga-g])+)', ' ' + field)
                if key:
                    key, mode = key.groups()
                    mode = mode.lower()[:3]
                    if mode not in s.offTab:
                        mode = 'maj'
                    fifths = s.sharpness.index(key) - s.offTab[mode]
                    if fifths >= 0:
                        s.keyAlts = dict(zip(accs[:fifths], fifths * ['1']))
                    else:
                        s.keyAlts = dict(zip(accs[fifths:], -fifths * ['-1']))
                elif field.startswith('none') or field == '':
                    fifths = 0
                    mode = 'maj'
                if alts:
                    alts = re.findall('[=^_][A-Ga-g]', alts.group(1))
                    alts = [(x[1], s.alterTab[x[0]]) for x in alts]
                    for step, alter in alts:
                        s.keyAlts[step.upper()] = alter
                    k = E.Element('key')
                    koctave = []
                    lowerCaseSteps = [step.upper() for step, alter in alts if step.islower()]
                    for step, alter in sorted(list(s.keyAlts.items())):
                        if alter == '0':
                            del s.keyAlts[step.upper()]
                            continue
                        addElemT(k, 'key-step', step.upper(), lev + 2)
                        addElemT(k, 'key-alter', alter, lev + 2)
                        koctave.append('5' if step in lowerCaseSteps else '4')
                    if koctave:
                        for oct in koctave:
                            e = E.Element('key-octave', number=oct)
                            addElem(k, e, lev + 2)
                        atts.append((2, k))
                elif mode:
                    k = E.Element('key')
                    addElemT(k, 'fifths', str(fifths), lev + 2)
                    addElemT(k, 'mode', s.modTab[mode], lev + 2)
                    atts.append((2, k))
                doClef(field)
            elif ftype == 'L':
                try:
                    s.unitLcur = lmap(int, field.split('/'))
                except:
                    s.unitLcur = (1, 8)
                if len(s.unitLcur) == 1 or s.unitLcur[1] not in s.typeMap:
                    info('L:%s is not allowed, 1/8 assumed' % field)
                    s.unitLcur = (1, 8)
            elif ftype == 'V':
                doClef(field)
            elif ftype == 'I':
                s.doField_I(ftype, field, instDir, addTrans)
            elif ftype == 'Q':
                s.doTempo(maat, field, lev)
            elif ftype == 'P':
                words = E.Element('rehearsal')
                words.set('font-weight', 'bold')
                words.text = field
                addDirection(maat, words, lev, gstaff, placement='above')
            elif ftype in 'TCOAZNGHRBDFSU':
                info('**illegal header field in body: %s, content: %s' % (ftype, field))
            else:
                info('unhandled field: %s, content: %s' % (ftype, field))
        if atts:
            att = E.Element('attributes')
            addElem(maat, att, lev)
            for _, att_elem in sorted(atts, key=lambda x: x[0]):
                addElem(att, att_elem, lev + 1)
        if s.diafret:
            other = E.Element('other-direction')
            other.text = 'diatonic fretting'
            addDirection(maat, other, lev, 0)

    def doTempo(s, maat, field, lev):
        gstaff = s.gStaffNums.get(s.vid, 0)
        t = re.search('(\\d)/(\\d\\d?)\\s*=\\s*(\\d[.\\d]*)|(\\d[.\\d]*)', field)
        rtxt = re.search('"([^"]*)"', field)
        if not t and (not rtxt):
            return
        elems = []
        if rtxt:
            num, den, upm = (1, 4, s.tempoMap.get(rtxt.group(1).lower().strip(), 120))
            words = E.Element('words')
            words.text = rtxt.group(1)
            elems.append((words, []))
        if t:
            try:
                if t.group(4):
                    num, den, upm = (1, s.unitLcur[1], float(t.group(4)))
                else:
                    num, den, upm = (int(t.group(1)), int(t.group(2)), float(t.group(3)))
            except:
                info('conversion error: %s' % field)
                return
            num, den = simplify(num, den)
            dotted, den_not = (1, den // 2) if num == 3 else (0, den)
            metro = E.Element('metronome')
            u = E.Element('beat-unit')
            u.text = s.typeMap.get(4 * den_not, 'quarter')
            pm = E.Element('per-minute')
            pm.text = ('%.2f' % upm).rstrip('0').rstrip('.')
            subelms = [u, E.Element('beat-unit-dot'), pm] if dotted else [u, pm]
            elems.append((metro, subelms))
        dir = addDirection(maat, elems, lev, gstaff, [], placement='above')
        if num != 1 and num != 3:
            info('in Q: numerator in %d/%d not supported' % (num, den))
        qpm = 4.0 * num * upm / den
        sound = E.Element('sound')
        sound.set('tempo', '%.2f' % qpm)
        addElem(dir, sound, lev + 1)

    def mkBarline(s, maat, loc, lev, style='', dir='', ending=''):
        b = E.Element('barline', location=loc)
        if style:
            addElemT(b, 'bar-style', style, lev + 1)
        if s.curVolta:
            end = E.Element('ending', number=s.curVolta, type='stop')
            s.curVolta = ''
            if loc == 'left':
                bp = E.Element('barline', location='right')
                addElem(bp, end, lev + 1)
                addElem(s.prevmsre, bp, lev)
            else:
                addElem(b, end, lev + 1)
        if ending:
            ending = ending.replace('-', ',')
            endtxt = ''
            if ending.startswith('"'):
                endtxt = ending.strip('"')
                ending = '33'
            end = E.Element('ending', number=ending, type='start')
            if endtxt:
                end.text = endtxt
            addElem(b, end, lev + 1)
            s.curVolta = ending
        if dir:
            r = E.Element('repeat', direction=dir)
            addElem(b, r, lev + 1)
        addElem(maat, b, lev)

    def doChordSym(s, maat, sym, lev):
        alterMap = {'#': '1', '=': '0', 'b': '-1'}
        rnt = sym.root.t
        chord = E.Element('harmony')
        addElem(maat, chord, lev)
        root = E.Element('root')
        addElem(chord, root, lev + 1)
        addElemT(root, 'root-step', rnt[0], lev + 2)
        if len(rnt) == 2:
            addElemT(root, 'root-alter', alterMap[rnt[1]], lev + 2)
        kind = s.chordTab.get(sym.kind.t[0], 'major') if sym.kind.t else 'major'
        addElemT(chord, 'kind', kind, lev + 1)
        if hasattr(sym, 'bass'):
            bnt = sym.bass.t
            bass = E.Element('bass')
            addElem(chord, bass, lev + 1)
            addElemT(bass, 'bass-step', bnt[0], lev + 2)
            if len(bnt) == 2:
                addElemT(bass, 'bass-alter', alterMap[bnt[1]], lev + 2)
        degs = getattr(sym, 'degree', '')
        if degs:
            if type(degs) != list_type:
                degs = [degs]
            for deg in degs:
                deg = deg.t[0]
                if deg[0] == '#':
                    alter = '1'
                    deg = deg[1:]
                elif deg[0] == 'b':
                    alter = '-1'
                    deg = deg[1:]
                else:
                    alter = '0'
                    deg = deg
                degree = E.Element('degree')
                addElem(chord, degree, lev + 1)
                addElemT(degree, 'degree-value', deg, lev + 2)
                addElemT(degree, 'degree-alter', alter, lev + 2)
                addElemT(degree, 'degree-type', 'add', lev + 2)

    def mkMeasure(s, i, t, lev, fieldmap={}):
        s.msreAlts = {}
        s.ntup, s.trem, s.intrem = (-1, 0, 0)
        s.acciatura = 0
        overlay = 0
        maat = E.Element('measure', number=str(i))
        if fieldmap:
            s.doFields(maat, fieldmap, lev + 1)
        if s.linebrk:
            e = E.Element('print')
            e.set('new-system', 'yes')
            addElem(maat, e, lev + 1)
            s.linebrk = 0
        for it, x in enumerate(t):
            if x.name == 'note' or x.name == 'rest':
                if x.dur.t[0] == 0:
                    x.dur.t = tuple([1, x.dur.t[1]])
                note = s.mkNote(x, lev + 1)
                addElem(maat, note, lev + 1)
            elif x.name == 'lbar':
                bar = x.t[0]
                if bar == '|' or bar == '[|':
                    pass
                elif ':' in bar:
                    volta = x.t[1] if len(x.t) == 2 else ''
                    s.mkBarline(maat, 'left', lev + 1, style='heavy-light', dir='forward', ending=volta)
                else:
                    s.mkBarline(maat, 'left', lev + 1, ending=bar)
            elif x.name == 'rbar':
                bar = x.t[0]
                if bar == '.|':
                    s.mkBarline(maat, 'right', lev + 1, style='dotted')
                elif ':' in bar:
                    s.mkBarline(maat, 'right', lev + 1, style='light-heavy', dir='backward')
                elif bar == '||':
                    s.mkBarline(maat, 'right', lev + 1, style='light-light')
                elif bar == '[|]' or bar == '[]':
                    s.mkBarline(maat, 'right', lev + 1, style='none')
                elif '[' in bar or ']' in bar:
                    s.mkBarline(maat, 'right', lev + 1, style='light-heavy')
                elif bar[0] == '&':
                    overlay = 1
            elif x.name == 'tup':
                if len(x.t) == 3:
                    n, into, nts = x.t
                elif len(x.t) == 2:
                    n, into, nts = x.t + [0]
                else:
                    n, into, nts = (x.t[0], 0, 0)
                if into == 0:
                    into = 3 if n in [2, 4, 8] else 2
                if nts == 0:
                    nts = n
                s.tmnum, s.tmden, s.ntup = (n, into, nts)
            elif x.name == 'deco':
                s.staffDecos(x.t, maat, lev + 1)
            elif x.name == 'text':
                pos, text = x.t[:2]
                place = 'above' if pos == '^' else 'below'
                words = E.Element('words')
                words.text = text
                gstaff = s.gStaffNums.get(s.vid, 0)
                addDirection(maat, words, lev + 1, gstaff, placement=place)
            elif x.name == 'inline':
                fieldtype, fieldval = (x.t[0], ' '.join(x.t[1:]))
                s.doFields(maat, {fieldtype: fieldval}, lev + 1)
            elif x.name == 'accia':
                s.acciatura = 1
            elif x.name == 'linebrk':
                s.supports_tag = 1
                if it > 0 and t[it - 1].name == 'lbar':
                    e = E.Element('print')
                    e.set('new-system', 'yes')
                    addElem(maat, e, lev + 1)
                else:
                    s.linebrk = 1
            elif x.name == 'chordsym':
                s.doChordSym(maat, x, lev + 1)
        s.stopBeams()
        s.prevmsre = maat
        return (maat, overlay)

    def mkPart(s, maten, id, lev, attrs, nstaves, rOpt):
        s.slurstack = {}
        s.glisnum = 0
        s.slidenum = 0
        s.unitLcur = s.unitL
        s.curVolta = ''
        s.lyrdash = {}
        s.linebrk = 0
        s.midprg = ['', '', '', '']
        s.gcue_on = 0
        s.gtrans = 0
        s.percVoice = 0
        s.curClef = ''
        s.nostems = 0
        s.tuning = s.tuningDef
        part = E.Element('part', id=id)
        s.overlayVnum = 0
        gstaff = s.gStaffNums.get(s.vid, 0)
        attrs_cpy = attrs.copy()
        if gstaff == 1:
            attrs_cpy['gstaff'] = nstaves
        if 'perc' in attrs_cpy.get('V', ''):
            del attrs_cpy['K']
        msre, overlay = s.mkMeasure(1, maten[0], lev + 1, attrs_cpy)
        addElem(part, msre, lev + 1)
        for i, maat in enumerate(maten[1:]):
            s.overlayVnum = s.overlayVnum + 1 if overlay else 0
            msre, next_overlay = s.mkMeasure(i + 2, maat, lev + 1)
            if overlay:
                mergePartMeasure(part, msre, s.overlayVnum, rOpt)
            else:
                addElem(part, msre, lev + 1)
            overlay = next_overlay
        return part

    def mkScorePart(s, id, vids_p, partAttr, lev):

        def mkInst(instId, vid, midchan, midprog, midnot, vol, pan, lev):
            si = E.Element('score-instrument', id=instId)
            pnm = partAttr.get(vid, [''])[0]
            addElemT(si, 'instrument-name', pnm or 'dummy', lev + 2)
            mi = E.Element('midi-instrument', id=instId)
            if midchan:
                addElemT(mi, 'midi-channel', midchan, lev + 2)
            if midprog:
                addElemT(mi, 'midi-program', str(int(midprog) + 1), lev + 2)
            if midnot:
                addElemT(mi, 'midi-unpitched', str(int(midnot) + 1), lev + 2)
            if vol:
                addElemT(mi, 'volume', '%.2f' % (int(vol) / 1.27), lev + 2)
            if pan:
                addElemT(mi, 'pan', '%.2f' % (int(pan) / 127.0 * 180 - 90), lev + 2)
            return (si, mi)
        naam, subnm, midprg = partAttr[id]
        sp = E.Element('score-part', id='P' + id)
        nm = E.Element('part-name')
        nm.text = naam
        addElem(sp, nm, lev + 1)
        snm = E.Element('part-abbreviation')
        snm.text = subnm
        if subnm:
            addElem(sp, snm, lev + 1)
        inst = []
        for instId, (pid, vid, chan, midprg, vol, pan) in sorted(s.midiInst.items()):
            midprg, midnot = ('0', midprg) if chan == '10' else (midprg, '')
            if pid == id:
                inst.append(mkInst(instId, vid, chan, midprg, midnot, vol, pan, lev))
        for si, mi in inst:
            addElem(sp, si, lev + 1)
        for si, mi in inst:
            addElem(sp, mi, lev + 1)
        return sp

    def mkPartlist(s, vids, partAttr, lev):

        def addPartGroup(sym, num):
            pg = E.Element('part-group', number=str(num), type='start')
            addElem(partlist, pg, lev + 1)
            addElemT(pg, 'group-symbol', sym, lev + 2)
            addElemT(pg, 'group-barline', 'yes', lev + 2)
        partlist = E.Element('part-list')
        g_num = 0
        for g in s.groups or vids:
            if g == '[':
                g_num += 1
                addPartGroup('bracket', g_num)
            elif g == '{':
                g_num += 1
                addPartGroup('brace', g_num)
            elif g in '}]':
                pg = E.Element('part-group', number=str(g_num), type='stop')
                addElem(partlist, pg, lev + 1)
                g_num -= 1
            else:
                if g not in vids:
                    continue
                sp = s.mkScorePart(g, vids, partAttr, lev + 1)
                addElem(partlist, sp, lev + 1)
        return partlist

    def doField_I(s, type, x, instDir, addTrans):

        def instChange(midchan, midprog):
            if midchan and midchan != s.midprg[0]:
                instDir('midi-channel', midchan, 'chan: %s')
            if midprog and midprog != s.midprg[1]:
                instDir('midi-program', str(int(midprog) + 1), 'prog: %s')

        def readPfmt(x, n):
            if not s.pageFmtAbc:
                s.pageFmtAbc = s.pageFmtDef
            ro = re.search('[^.\\d]*([\\d.]+)\\s*(cm|in|pt)?', x)
            if ro:
                x, unit = ro.groups()
                u = {'cm': 10.0, 'in': 25.4, 'pt': 25.4 / 72}[unit] if unit else 1.0
                s.pageFmtAbc[n] = float(x) * u
            else:
                info('error in page format: %s' % x)

        def readPercMap(x):

            def getMidNum(sndnm):
                pnms = sndnm.split('-')
                ps = s.percsnd[:]
                _f = lambda ip, xs, pnm: ip < len(xs) and xs[ip].find(pnm) > -1
                for ip, pnm in enumerate(pnms):
                    ps = [(nm, mnum) for nm, mnum in ps if _f(ip, nm.split('-'), pnm)]
                    if len(ps) <= 1:
                        break
                if len(ps) == 0:
                    info('drum sound: %s not found' % sndnm)
                    return '38'
                return ps[0][1]

            def midiVal(acc, step, oct):
                oct = (4 if step.upper() == step else 5) + int(oct)
                return oct * 12 + [0, 2, 4, 5, 7, 9, 11]['CDEFGAB'.index(step.upper())] + {'^': 1, '_': -1, '=': 0}.get(acc, 0) + 12
            p0, p1, p2, p3, p4 = abc_percmap.parseString(x).asList()
            acc, astep, aoct = p1
            nstep, noct = (astep, aoct) if p2 == '*' else p2
            if p3 == '*':
                midi = str(midiVal(acc, astep, aoct))
            elif isinstance(p3, list_type):
                midi = str(midiVal(p3[0], p3[1], p3[2]))
            elif isinstance(p3, int_type):
                midi = str(p3)
            else:
                midi = getMidNum(p3.lower())
            head = re.sub('(.)-([^x])', '\\1 \\2', p4)
            s.percMap[s.pid, acc + astep, aoct] = (nstep, noct, midi, head)
        if x.startswith('score') or x.startswith('staves'):
            s.staveDefs += [x]
        elif x.startswith('staffwidth'):
            info('skipped I-field: %s' % x)
        elif x.startswith('staff'):
            r1 = re.search('staff *([+-]?)(\\d)', x)
            if r1:
                sign = r1.group(1)
                num = int(r1.group(2))
                gstaff = s.gStaffNums.get(s.vid, 0)
                if sign:
                    num = sign == '-' and gstaff - num or gstaff + num
                else:
                    try:
                        vabc = s.staves[num - 1][0]
                    except:
                        vabc = 0
                        info('abc staff %s does not exist' % num)
                    num = s.gStaffNumsOrg.get(vabc, 0)
                if gstaff and num > 0 and (num <= s.gNstaves[s.vid]):
                    s.gStaffNums[s.vid] = num
                else:
                    info('could not relocate to staff: %s' % r1.group())
            else:
                info('not a valid staff redirection: %s' % x)
        elif x.startswith('scale'):
            readPfmt(x, 0)
        elif x.startswith('pageheight'):
            readPfmt(x, 1)
        elif x.startswith('pagewidth'):
            readPfmt(x, 2)
        elif x.startswith('leftmargin'):
            readPfmt(x, 3)
        elif x.startswith('rightmargin'):
            readPfmt(x, 4)
        elif x.startswith('topmargin'):
            readPfmt(x, 5)
        elif x.startswith('botmargin'):
            readPfmt(x, 6)
        elif x.startswith('MIDI') or x.startswith('midi'):
            r1 = re.search('program *(\\d*) +(\\d+)', x)
            r2 = re.search('channel *(\\d+)', x)
            r3 = re.search("drummap\\s+([_=^]*)([A-Ga-g])([,']*)\\s+(\\d+)", x)
            r4 = re.search('control *(\\d+) +(\\d+)', x)
            ch_nw, prg_nw, vol_nw, pan_nw = ('', '', '', '')
            if r1:
                ch_nw, prg_nw = r1.groups()
            if r2:
                ch_nw = r2.group(1)
            if r4:
                cnum, cval = r4.groups()
                if cnum == '7':
                    vol_nw = cval
                if cnum == '10':
                    pan_nw = cval
            if r1 or r2 or r4:
                ch = ch_nw or s.midprg[0]
                prg = prg_nw or s.midprg[1]
                vol = vol_nw or s.midprg[2]
                pan = pan_nw or s.midprg[3]
                instId = 'I%s-%s' % (s.pid, s.vid)
                if instId in s.midiInst:
                    instChange(ch, prg)
                s.midprg = [ch, prg, vol, pan]
            if r3:
                acc, step, oct, midi = r3.groups()
                oct = -len(oct) if ',' in x else len(oct)
                notehead = 'x' if acc == '^' else 'circle-x' if acc == '_' else 'normal'
                s.percMap[s.pid, acc + step, oct] = (step, oct, midi, notehead)
            r = re.search('transpose[^-\\d]*(-?\\d+)', x)
            if r:
                addTrans(r.group(1))
        elif x.startswith('percmap'):
            readPercMap(x)
            s.pMapFound = 1
        else:
            info('skipped I-field: %s' % x)

    def parseStaveDef(s, vdefs):
        for vid in vdefs:
            s.vcepid[vid] = vid
        if not s.staveDefs:
            return vdefs
        for x in s.staveDefs[1:]:
            info('%%%%%s dropped, multiple stave mappings not supported' % x)
        x = s.staveDefs[0]
        score = abc_scoredef.parseString(x)[0]
        f = lambda x: type(x) == uni_type and [x] or x
        s.staves = lmap(f, mkStaves(score, vdefs))
        s.grands = lmap(f, mkGrand(score, vdefs))
        s.groups = mkGroups(score)
        vce_groups = [vids for vids in s.staves if len(vids) > 1]
        d = {}
        for vgr in vce_groups:
            d[vgr[0]] = vgr
        for gstaff in s.grands:
            if len(gstaff) == 1:
                continue
            for v, stf_num in zip(gstaff, range(1, len(gstaff) + 1)):
                for vx in d.get(v, [v]):
                    s.gStaffNums[vx] = stf_num
                    s.gNstaves[vx] = len(gstaff)
        s.gStaffNumsOrg = s.gStaffNums.copy()
        for xmlpart in s.grands:
            pid = xmlpart[0]
            vces = [v for stf in xmlpart for v in d.get(stf, [stf])]
            for v in vces:
                s.vcepid[v] = pid
        return vdefs

    def voiceNamesAndMaps(s, ps):
        vdefs = {}
        for vid, vcedef, vce in ps:
            pname, psubnm = ('', '')
            if not vcedef:
                vdefs[vid] = (pname, psubnm, '')
            else:
                if vid != vcedef.t[1]:
                    info('voice ids unequal: %s (reg-ex) != %s (grammar)' % (vid, vcedef.t[1]))
                rn = re.search('(?:name|nm)="([^"]*)"', vcedef.t[2])
                if rn:
                    pname = rn.group(1)
                rn = re.search('(?:subname|snm|sname)="([^"]*)"', vcedef.t[2])
                if rn:
                    psubnm = rn.group(1)
                vcedef.t[2] = vcedef.t[2].replace('"%s"' % pname, '""').replace('"%s"' % psubnm, '""')
                vdefs[vid] = (pname, psubnm, vcedef.t[2])
            xs = [pObj.t[1] for maat in vce for pObj in maat if pObj.name == 'inline']
            s.staveDefs += [x.replace('%5d', ']') for x in xs if x.startswith('score') or x.startswith('staves')]
        return vdefs

    def doHeaderField(s, fld, attrmap):
        type, value = (fld.t[0], fld.t[1].replace('%5d', ']'))
        if not value:
            return
        if type == 'M':
            attrmap[type] = value
        elif type == 'L':
            try:
                s.unitL = lmap(int, fld.t[1].split('/'))
            except:
                info('illegal unit length:%s, 1/8 assumed' % fld.t[1])
                s.unitL = (1, 8)
            if len(s.unitL) == 1 or s.unitL[1] not in s.typeMap:
                info('L:%s is not allowed, 1/8 assumed' % fld.t[1])
                s.unitL = (1, 8)
        elif type == 'K':
            attrmap[type] = value
        elif type == 'T':
            s.title = s.title + '\n' + value if s.title else value
        elif type == 'U':
            sym = fld.t[2].strip('!+')
            s.usrSyms[value] = sym
        elif type == 'I':
            s.doField_I(type, value, lambda x, y, z: 0, lambda x: 0)
        elif type == 'Q':
            attrmap[type] = value
        elif type in 'CRZNOAGHBDFSP':
            type = s.metaMap.get(type, type)
            c = s.metadata.get(type, '')
            s.metadata[type] = c + '\n' + value if c else value
        else:
            info('skipped header: %s' % fld)

    def mkIdentification(s, score, lev):
        if s.title:
            xs = s.title.split('\n')
            ys = '\n'.join(xs[1:])
            w = E.Element('work')
            addElem(score, w, lev + 1)
            if ys:
                addElemT(w, 'work-number', ys, lev + 2)
            addElemT(w, 'work-title', xs[0], lev + 2)
        ident = E.Element('identification')
        addElem(score, ident, lev + 1)
        for mtype, mval in s.metadata.items():
            if mtype in s.metaTypes and mtype != 'rights':
                c = E.Element('creator', type=mtype)
                c.text = mval
                addElem(ident, c, lev + 2)
        if 'rights' in s.metadata:
            c = addElemT(ident, 'rights', s.metadata['rights'], lev + 2)
        encoding = E.Element('encoding')
        addElem(ident, encoding, lev + 2)
        encoder = E.Element('encoder')
        encoder.text = 'abc2xml version %d' % VERSION
        addElem(encoding, encoder, lev + 3)
        if s.supports_tag:
            suports = E.Element('supports', attribute='new-system', element='print', type='yes', value='yes')
            addElem(encoding, suports, lev + 3)
        encodingDate = E.Element('encoding-date')
        encodingDate.text = str(datetime.date.today())
        addElem(encoding, encodingDate, lev + 3)
        s.addMeta(ident, lev + 2)

    def mkDefaults(s, score, lev):
        if s.pageFmtCmd:
            s.pageFmtAbc = s.pageFmtCmd
        if not s.pageFmtAbc:
            return
        abcScale, h, w, l, r, t, b = s.pageFmtAbc
        space = abcScale * 2.117
        mils = 4 * space
        scale = 40.0 / mils
        dflts = E.Element('defaults')
        addElem(score, dflts, lev)
        scaling = E.Element('scaling')
        addElem(dflts, scaling, lev + 1)
        addElemT(scaling, 'millimeters', '%g' % mils, lev + 2)
        addElemT(scaling, 'tenths', '40', lev + 2)
        layout = E.Element('page-layout')
        addElem(dflts, layout, lev + 1)
        addElemT(layout, 'page-height', '%g' % (h * scale), lev + 2)
        addElemT(layout, 'page-width', '%g' % (w * scale), lev + 2)
        margins = E.Element('page-margins', type='both')
        addElem(layout, margins, lev + 2)
        addElemT(margins, 'left-margin', '%g' % (l * scale), lev + 3)
        addElemT(margins, 'right-margin', '%g' % (r * scale), lev + 3)
        addElemT(margins, 'top-margin', '%g' % (t * scale), lev + 3)
        addElemT(margins, 'bottom-margin', '%g' % (b * scale), lev + 3)

    def addMeta(s, parent, lev):
        misc = E.Element('miscellaneous')
        mf = 0
        for mtype, mval in sorted(s.metadata.items()):
            if mtype == 'S':
                addElemT(parent, 'source', mval, lev)
            elif mtype in s.metaTypes:
                continue
            else:
                mf = E.Element('miscellaneous-field', name=s.metaTab[mtype])
                mf.text = mval
                addElem(misc, mf, lev + 1)
        if mf != 0:
            addElem(parent, misc, lev)

    def parse(s, abc_string, rOpt=False, bOpt=False, fOpt=False):
        abctext = abc_string.replace('[I:staff ', '[I:staff')
        s.reset(fOpt)
        header, voices = splitHeaderVoices(abctext)
        ps = []
        try:
            lbrk_insert = 0 if re.search('I:linebreak\\s*([!$]|none)|I:continueall\\s*(1|true)', header) else bOpt
            hs = abc_header.parseString(header) if header else ''
            for id, voice in voices:
                if lbrk_insert:
                    r1 = re.compile('\\[[wA-Z]:[^]]*\\]')
                    has_abc = lambda x: r1.sub('', x).strip()
                    voice = '\n'.join([balk.rstrip('$!') + '$' if has_abc(balk) else balk for balk in voice.splitlines()])
                prevLeftBar = None
                s.orderChords = s.fOpt and ('tab' in voice[:200] or [x for x in hs if x.t[0] == 'K' and 'tab' in x.t[1]])
                vce = abc_voice.parseString(voice).asList()
                lyr_notes = []
                for m in vce:
                    for e in m:
                        if e.name == 'lyr_blk':
                            lyr = [line.objs for line in e.objs]
                            alignLyr(lyr_notes, lyr)
                            lyr_notes = []
                        else:
                            lyr_notes.append(e)
                if not vce:
                    vce = [[pObj('inline', ['I', 'empty voice'])]]
                if prevLeftBar:
                    vce[0].insert(0, prevLeftBar)
                    prevLeftBar = None
                if vce[-1] and vce[-1][-1].name == 'lbar':
                    prevLeftBar = vce[-1][-1]
                    if len(vce) > 1:
                        del vce[-1]
                vcelyr = vce
                elem1 = vcelyr[0][0]
                if elem1.name == 'inline' and elem1.t[0] == 'V':
                    voicedef = elem1
                    del vcelyr[0][0]
                else:
                    voicedef = ''
                ps.append((id, voicedef, vcelyr))
        except ParseException as err:
            if err.loc > 40:
                err.pstr = err.pstr[err.loc - 40:err.loc + 40]
                err.loc = 40
            xs = err.line[err.col - 1:]
            info(err.line, warn=0)
            info((err.col - 1) * '-' + '^', warn=0)
            if re.search('\\[U:', xs):
                info('Error: illegal user defined symbol: %s' % xs[1:], warn=0)
            elif re.search('\\[[OAPZNGHRBDFSXTCIU]:', xs):
                info('Error: header-only field %s appears after K:' % xs[1:], warn=0)
            else:
                info('Syntax error at column %d' % err.col, warn=0)
            raise
        score = E.Element('score-partwise')
        attrmap = {'Div': str(s.divisions), 'K': 'C treble', 'M': '4/4'}
        for res in hs:
            if res.name == 'field':
                s.doHeaderField(res, attrmap)
            else:
                info('unexpected header item: %s' % res)
        vdefs = s.voiceNamesAndMaps(ps)
        vdefs = s.parseStaveDef(vdefs)
        lev = 0
        vids, parts, partAttr = ([], [], {})
        s.strAlloc = stringAlloc()
        for vid, _, vce in ps:
            pname, psubnm, voicedef = vdefs[vid]
            attrmap['V'] = voicedef
            pid = 'P%s' % vid
            s.vid = vid
            s.pid = s.vcepid[s.vid]
            s.gTime = (0, 0)
            s.strAlloc.beginZoek()
            part = s.mkPart(vce, pid, lev + 1, attrmap, s.gNstaves.get(vid, 0), rOpt)
            if 'Q' in attrmap:
                del attrmap['Q']
            parts.append(part)
            vids.append(vid)
            partAttr[vid] = (pname, psubnm, s.midprg)
            if s.midprg != ['', '', '', ''] and (not s.percVoice):
                instId = 'I%s-%s' % (s.pid, s.vid)
                if instId not in s.midiInst:
                    s.midiInst[instId] = (s.pid, s.vid, s.midprg[0], s.midprg[1], s.midprg[2], s.midprg[3])
        parts, vidsnew = mergeParts(parts, vids, s.staves, rOpt)
        parts, vidsnew = mergeParts(parts, vidsnew, s.grands, rOpt, 1)
        reduceMids(parts, vidsnew, s.midiInst)
        s.mkIdentification(score, lev)
        s.mkDefaults(score, lev + 1)
        partlist = s.mkPartlist(vids, partAttr, lev + 1)
        addElem(score, partlist, lev + 1)
        for ip, part in enumerate(parts):
            addElem(score, part, lev + 1)
        return score