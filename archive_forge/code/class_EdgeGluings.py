import snappy
import snappy.snap.t3mlite as t3m
import snappy.snap.peripheral as peripheral
from sage.all import ZZ, QQ, GF, gcd, PolynomialRing, cyclotomic_polynomial
class EdgeGluings:

    def __init__(self, gen_obs_class):
        assert gen_obs_class._N == 2
        M = gen_obs_class._manifold
        n = M.num_tetrahedra()
        gluings = t3m.files.read_SnapPea_file(data=M._to_string())
        primary_faces = [parse_ptolemy_face(x[2]) for x in M._ptolemy_equations_identified_face_classes()]
        ptolemy_idents = M._ptolemy_equations_identified_coordinates(2, gen_obs_class.H2_class)
        self.edge_gluings = edge_gluings = dict()
        for i, (tet0, face0) in enumerate(primary_faces):
            for j in range(3):
                orient_sign, obs_contrib, edge_var_0, edge_var_1 = ptolemy_idents[3 * i + j]
                sign = orient_sign * (-1) ** obs_contrib
                tet0alt, edge0 = parse_ptolemy_edge(edge_var_0)
                tet1, edge1 = parse_ptolemy_edge(edge_var_1)
                perm = gluings[tet0][1][face0]
                face1 = perm[face0]
                edge_gluings[tet0, face0, edge0] = [(tet1, face1, edge1), sign]
                edge_gluings[tet1, face1, edge1] = [(tet0, face0, edge0), sign]
                assert tet0 == tet0alt
                assert tet1 == gluings[tet0][0][face0]
                a, b = edge0
                c, d = edge1
                if perm[a] == c and perm[b] == d:
                    assert orient_sign == 1
                else:
                    assert perm[a] == d and perm[b] == c and (orient_sign == -1)

    def __getitem__(self, index):
        return self.edge_gluings[index]