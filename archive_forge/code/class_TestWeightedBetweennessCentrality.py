import pytest
import networkx as nx
class TestWeightedBetweennessCentrality:

    def test_K5(self):
        """Weighted betweenness centrality: K5"""
        G = nx.complete_graph(5)
        b = nx.betweenness_centrality(G, weight='weight', normalized=False)
        b_answer = {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0}
        for n in sorted(G):
            assert b[n] == pytest.approx(b_answer[n], abs=1e-07)

    def test_P3_normalized(self):
        """Weighted betweenness centrality: P3 normalized"""
        G = nx.path_graph(3)
        b = nx.betweenness_centrality(G, weight='weight', normalized=True)
        b_answer = {0: 0.0, 1: 1.0, 2: 0.0}
        for n in sorted(G):
            assert b[n] == pytest.approx(b_answer[n], abs=1e-07)

    def test_P3(self):
        """Weighted betweenness centrality: P3"""
        G = nx.path_graph(3)
        b_answer = {0: 0.0, 1: 1.0, 2: 0.0}
        b = nx.betweenness_centrality(G, weight='weight', normalized=False)
        for n in sorted(G):
            assert b[n] == pytest.approx(b_answer[n], abs=1e-07)

    def test_krackhardt_kite_graph(self):
        """Weighted betweenness centrality: Krackhardt kite graph"""
        G = nx.krackhardt_kite_graph()
        b_answer = {0: 1.667, 1: 1.667, 2: 0.0, 3: 7.333, 4: 0.0, 5: 16.667, 6: 16.667, 7: 28.0, 8: 16.0, 9: 0.0}
        for b in b_answer:
            b_answer[b] /= 2
        b = nx.betweenness_centrality(G, weight='weight', normalized=False)
        for n in sorted(G):
            assert b[n] == pytest.approx(b_answer[n], abs=0.001)

    def test_krackhardt_kite_graph_normalized(self):
        """Weighted betweenness centrality:
        Krackhardt kite graph normalized
        """
        G = nx.krackhardt_kite_graph()
        b_answer = {0: 0.023, 1: 0.023, 2: 0.0, 3: 0.102, 4: 0.0, 5: 0.231, 6: 0.231, 7: 0.389, 8: 0.222, 9: 0.0}
        b = nx.betweenness_centrality(G, weight='weight', normalized=True)
        for n in sorted(G):
            assert b[n] == pytest.approx(b_answer[n], abs=0.001)

    def test_florentine_families_graph(self):
        """Weighted betweenness centrality:
        Florentine families graph"""
        G = nx.florentine_families_graph()
        b_answer = {'Acciaiuoli': 0.0, 'Albizzi': 0.212, 'Barbadori': 0.093, 'Bischeri': 0.104, 'Castellani': 0.055, 'Ginori': 0.0, 'Guadagni': 0.255, 'Lamberteschi': 0.0, 'Medici': 0.522, 'Pazzi': 0.0, 'Peruzzi': 0.022, 'Ridolfi': 0.114, 'Salviati': 0.143, 'Strozzi': 0.103, 'Tornabuoni': 0.092}
        b = nx.betweenness_centrality(G, weight='weight', normalized=True)
        for n in sorted(G):
            assert b[n] == pytest.approx(b_answer[n], abs=0.001)

    def test_les_miserables_graph(self):
        """Weighted betweenness centrality: Les Miserables graph"""
        G = nx.les_miserables_graph()
        b_answer = {'Napoleon': 0.0, 'Myriel': 0.177, 'MlleBaptistine': 0.0, 'MmeMagloire': 0.0, 'CountessDeLo': 0.0, 'Geborand': 0.0, 'Champtercier': 0.0, 'Cravatte': 0.0, 'Count': 0.0, 'OldMan': 0.0, 'Valjean': 0.454, 'Labarre': 0.0, 'Marguerite': 0.009, 'MmeDeR': 0.0, 'Isabeau': 0.0, 'Gervais': 0.0, 'Listolier': 0.0, 'Tholomyes': 0.066, 'Fameuil': 0.0, 'Blacheville': 0.0, 'Favourite': 0.0, 'Dahlia': 0.0, 'Zephine': 0.0, 'Fantine': 0.114, 'MmeThenardier': 0.046, 'Thenardier': 0.129, 'Cosette': 0.075, 'Javert': 0.193, 'Fauchelevent': 0.026, 'Bamatabois': 0.08, 'Perpetue': 0.0, 'Simplice': 0.001, 'Scaufflaire': 0.0, 'Woman1': 0.0, 'Judge': 0.0, 'Champmathieu': 0.0, 'Brevet': 0.0, 'Chenildieu': 0.0, 'Cochepaille': 0.0, 'Pontmercy': 0.023, 'Boulatruelle': 0.0, 'Eponine': 0.023, 'Anzelma': 0.0, 'Woman2': 0.0, 'MotherInnocent': 0.0, 'Gribier': 0.0, 'MmeBurgon': 0.026, 'Jondrette': 0.0, 'Gavroche': 0.285, 'Gillenormand': 0.024, 'Magnon': 0.005, 'MlleGillenormand': 0.036, 'MmePontmercy': 0.005, 'MlleVaubois': 0.0, 'LtGillenormand': 0.015, 'Marius': 0.072, 'BaronessT': 0.004, 'Mabeuf': 0.089, 'Enjolras': 0.003, 'Combeferre': 0.0, 'Prouvaire': 0.0, 'Feuilly': 0.004, 'Courfeyrac': 0.001, 'Bahorel': 0.007, 'Bossuet': 0.028, 'Joly': 0.0, 'Grantaire': 0.036, 'MotherPlutarch': 0.0, 'Gueulemer': 0.025, 'Babet': 0.015, 'Claquesous': 0.042, 'Montparnasse': 0.05, 'Toussaint': 0.011, 'Child1': 0.0, 'Child2': 0.0, 'Brujon': 0.002, 'MmeHucheloup': 0.034}
        b = nx.betweenness_centrality(G, weight='weight', normalized=True)
        for n in sorted(G):
            assert b[n] == pytest.approx(b_answer[n], abs=0.001)

    def test_ladder_graph(self):
        """Weighted betweenness centrality: Ladder graph"""
        G = nx.Graph()
        G.add_edges_from([(0, 1), (0, 2), (1, 3), (2, 3), (2, 4), (4, 5), (3, 5)])
        b_answer = {0: 1.667, 1: 1.667, 2: 6.667, 3: 6.667, 4: 1.667, 5: 1.667}
        for b in b_answer:
            b_answer[b] /= 2
        b = nx.betweenness_centrality(G, weight='weight', normalized=False)
        for n in sorted(G):
            assert b[n] == pytest.approx(b_answer[n], abs=0.001)

    def test_G(self):
        """Weighted betweenness centrality: G"""
        G = weighted_G()
        b_answer = {0: 2.0, 1: 0.0, 2: 4.0, 3: 3.0, 4: 4.0, 5: 0.0}
        b = nx.betweenness_centrality(G, weight='weight', normalized=False)
        for n in sorted(G):
            assert b[n] == pytest.approx(b_answer[n], abs=1e-07)

    def test_G2(self):
        """Weighted betweenness centrality: G2"""
        G = nx.DiGraph()
        G.add_weighted_edges_from([('s', 'u', 10), ('s', 'x', 5), ('u', 'v', 1), ('u', 'x', 2), ('v', 'y', 1), ('x', 'u', 3), ('x', 'v', 5), ('x', 'y', 2), ('y', 's', 7), ('y', 'v', 6)])
        b_answer = {'y': 5.0, 'x': 5.0, 's': 4.0, 'u': 2.0, 'v': 2.0}
        b = nx.betweenness_centrality(G, weight='weight', normalized=False)
        for n in sorted(G):
            assert b[n] == pytest.approx(b_answer[n], abs=1e-07)

    def test_G3(self):
        """Weighted betweenness centrality: G3"""
        G = nx.MultiGraph(weighted_G())
        es = list(G.edges(data=True))[::2]
        G.add_edges_from(es)
        b_answer = {0: 2.0, 1: 0.0, 2: 4.0, 3: 3.0, 4: 4.0, 5: 0.0}
        b = nx.betweenness_centrality(G, weight='weight', normalized=False)
        for n in sorted(G):
            assert b[n] == pytest.approx(b_answer[n], abs=1e-07)

    def test_G4(self):
        """Weighted betweenness centrality: G4"""
        G = nx.MultiDiGraph()
        G.add_weighted_edges_from([('s', 'u', 10), ('s', 'x', 5), ('s', 'x', 6), ('u', 'v', 1), ('u', 'x', 2), ('v', 'y', 1), ('v', 'y', 1), ('x', 'u', 3), ('x', 'v', 5), ('x', 'y', 2), ('x', 'y', 3), ('y', 's', 7), ('y', 'v', 6), ('y', 'v', 6)])
        b_answer = {'y': 5.0, 'x': 5.0, 's': 4.0, 'u': 2.0, 'v': 2.0}
        b = nx.betweenness_centrality(G, weight='weight', normalized=False)
        for n in sorted(G):
            assert b[n] == pytest.approx(b_answer[n], abs=1e-07)