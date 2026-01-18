from reportlab.lib import colors
from reportlab.graphics.shapes import Rect, Drawing, Group, String
from reportlab.graphics.charts.textlabels import Label
from reportlab.graphics.widgetbase import Widget
class EventCalendar(Widget):

    def __init__(self):
        self.x = 0
        self.y = 0
        self.width = 300
        self.height = 150
        self.timeColWidth = None
        self.trackRowHeight = 20
        self.data = []
        self.trackNames = None
        self.startTime = None
        self.endTime = None
        self.day = 0
        self._talksVisible = []
        self._startTime = None
        self._endTime = None
        self._trackCount = 0
        self._colWidths = []
        self._colLeftEdges = []

    def computeSize(self):
        """Called at start of draw.  Sets various column widths"""
        self._talksVisible = self.getRelevantTalks(self.data)
        self._trackCount = len(self.getAllTracks())
        self.computeStartAndEndTimes()
        self._colLeftEdges = [self.x]
        if self.timeColWidth is None:
            w = self.width / (1 + self._trackCount)
            self._colWidths = [w] * (1 + self._trackCount)
            for i in range(self._trackCount):
                self._colLeftEdges.append(self._colLeftEdges[-1] + w)
        else:
            self._colWidths = [self.timeColWidth]
            w = (self.width - self.timeColWidth) / self._trackCount
            for i in range(self._trackCount):
                self._colWidths.append(w)
                self._colLeftEdges.append(self._colLeftEdges[-1] + w)

    def computeStartAndEndTimes(self):
        """Work out first and last times to display"""
        if self.startTime:
            self._startTime = self.startTime
        else:
            for title, speaker, trackId, day, start, duration in self._talksVisible:
                if self._startTime is None:
                    self._startTime = start
                elif start < self._startTime:
                    self._startTime = start
        if self.endTime:
            self._endTime = self.endTime
        else:
            for title, speaker, trackId, day, start, duration in self._talksVisible:
                if self._endTime is None:
                    self._endTime = start + duration
                elif start + duration > self._endTime:
                    self._endTime = start + duration

    def getAllTracks(self):
        tracks = []
        for title, speaker, trackId, day, hours, duration in self.data:
            if trackId is not None:
                if trackId not in tracks:
                    tracks.append(trackId)
        tracks.sort()
        return tracks

    def getRelevantTalks(self, talkList):
        """Scans for tracks actually used"""
        used = []
        for talk in talkList:
            title, speaker, trackId, day, hours, duration = talk
            assert trackId != 0, 'trackId must be None or 1,2,3... zero not allowed!'
            if day == self.day:
                if (self.startTime is None or hours + duration >= self.startTime) and (self.endTime is None or hours <= self.endTime):
                    used.append(talk)
        return used

    def scaleTime(self, theTime):
        """Return y-value corresponding to times given"""
        axisHeight = self.height - self.trackRowHeight
        proportionUp = (theTime - self._startTime) / (self._endTime - self._startTime)
        y = self.y + axisHeight - axisHeight * proportionUp
        return y

    def getTalkRect(self, startTime, duration, trackId, text):
        """Return shapes for a specific talk"""
        g = Group()
        y_bottom = self.scaleTime(startTime + duration)
        y_top = self.scaleTime(startTime)
        y_height = y_top - y_bottom
        if trackId is None:
            x = self._colLeftEdges[1]
            width = self.width - self._colWidths[0]
        else:
            x = self._colLeftEdges[trackId]
            width = self._colWidths[trackId]
        lab = Label()
        lab.setText(text)
        lab.setOrigin(x + 0.5 * width, y_bottom + 0.5 * y_height)
        lab.boxAnchor = 'c'
        lab.width = width
        lab.height = y_height
        lab.fontSize = 6
        r = Rect(x, y_bottom, width, y_height, fillColor=colors.cyan)
        g.add(r)
        g.add(lab)
        return g

    def draw(self):
        self.computeSize()
        g = Group()
        g.add(Rect(self.x, self.y, self._colWidths[0], self.height - self.trackRowHeight, fillColor=colors.cornsilk))
        x = self.x + self._colWidths[0]
        y = self.y + self.height - self.trackRowHeight
        for trk in range(self._trackCount):
            wid = self._colWidths[trk + 1]
            r = Rect(x, y, wid, self.trackRowHeight, fillColor=colors.yellow)
            s = String(x + 0.5 * wid, y, 'Track %d' % trk, align='middle')
            g.add(r)
            g.add(s)
            x = x + wid
        for talk in self._talksVisible:
            title, speaker, trackId, day, start, duration = talk
            r = self.getTalkRect(start, duration, trackId, title + '\n' + speaker)
            g.add(r)
        return g