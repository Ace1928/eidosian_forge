class CountBads:
    """Helper to report anomalies in the media_player states seen when playing
     a sample.

        - provides .anomalies_description, a dict <anomaly>: <description>
        - calling .count_bads(recorded_events) will return a dict of
          anomaly: <count times anomaly detected>
        - preprocessing: ad-hoc prefiltering the events stream for noise reduction
     """

    def __init__(self, events_definition=mp_events, bads=mp_bads):
        self.events_definition = events_definition
        self.bads = bads
        self.anomalies_description = self.build_anomalies_description()

    def build_anomalies_description(self):
        """builds descriptions for the anomalies"""
        d = self.events_definition
        anomalies_description = {evname: d[evname]['desc'] for evname in self.bads}
        anomalies_description['scheduling_in_past'] = 'Scheduling in the past'
        return anomalies_description

    def preprocessing(self, recorded_events):
        """
        I see all recordings ending with some potential anomalies in the few
        frames just before the '>>> play ends'; visually the play is perfect so
        I assume they are false positives if just at EOF. Deleting the offending
        events (only if near EOL) to reduce noise in summarize.py
        """
        recorded_events = list(recorded_events)
        if len(recorded_events) > 9 and recorded_events[-2][0] == 'p.P.ut.1.7' and (recorded_events[-6][0] == 'p.P.ut.1.7') and (recorded_events[-10][0] == 'p.P.ut.1.7'):
            del recorded_events[-10]
            del recorded_events[-6]
            del recorded_events[-2]
        elif len(recorded_events) > 6 and recorded_events[-2][0] == 'p.P.ut.1.7' and (recorded_events[-6][0] == 'p.P.ut.1.7'):
            del recorded_events[-6]
            del recorded_events[-2]
        elif len(recorded_events) > 2 and recorded_events[-2][0] == 'p.P.ut.1.7':
            del recorded_events[-2]
        return recorded_events

    def count_bads(self, recorded_events):
        """returns counts of anomalies as a dict of anomaly: count

        recorded_events: media_player events recorded while playing a sample

        Notice that 'counters' has one more key than 'bads': "scheduling_in_past"
        """
        recorded_events = self.preprocessing(recorded_events)
        counters = {k: 0 for k in self.bads}
        cnt_scheduling_in_past = 0
        mp_states = MediaPlayerStateIterator(recorded_events, self.events_definition)
        for st in mp_states:
            evname = st['evname']
            if evname in counters:
                counters[evname] += 1
            elif 'p.P.ut.1.9' and st['rescheduling_time'] is not None and (st['rescheduling_time'] < 0):
                cnt_scheduling_in_past += 1
        counters['scheduling_in_past'] = cnt_scheduling_in_past
        return counters