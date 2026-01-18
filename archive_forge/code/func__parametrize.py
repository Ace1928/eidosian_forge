from fontTools.misc.transform import Identity, Scale
from math import atan2, ceil, cos, fabs, isfinite, pi, radians, sin, sqrt, tan
def _parametrize(self):
    rx = fabs(self.rx)
    ry = fabs(self.ry)
    if not (rx and ry):
        return False
    if self.target_point == self.current_point:
        return False
    mid_point_distance = (self.current_point - self.target_point) * 0.5
    point_transform = Identity.rotate(-self.angle)
    transformed_mid_point = _map_point(point_transform, mid_point_distance)
    square_rx = rx * rx
    square_ry = ry * ry
    square_x = transformed_mid_point.real * transformed_mid_point.real
    square_y = transformed_mid_point.imag * transformed_mid_point.imag
    radii_scale = square_x / square_rx + square_y / square_ry
    if radii_scale > 1:
        rx *= sqrt(radii_scale)
        ry *= sqrt(radii_scale)
        self.rx, self.ry = (rx, ry)
    point_transform = Scale(1 / rx, 1 / ry).rotate(-self.angle)
    point1 = _map_point(point_transform, self.current_point)
    point2 = _map_point(point_transform, self.target_point)
    delta = point2 - point1
    d = delta.real * delta.real + delta.imag * delta.imag
    scale_factor_squared = max(1 / d - 0.25, 0.0)
    scale_factor = sqrt(scale_factor_squared)
    if self.sweep == self.large:
        scale_factor = -scale_factor
    delta *= scale_factor
    center_point = (point1 + point2) * 0.5
    center_point += complex(-delta.imag, delta.real)
    point1 -= center_point
    point2 -= center_point
    theta1 = atan2(point1.imag, point1.real)
    theta2 = atan2(point2.imag, point2.real)
    theta_arc = theta2 - theta1
    if theta_arc < 0 and self.sweep:
        theta_arc += TWO_PI
    elif theta_arc > 0 and (not self.sweep):
        theta_arc -= TWO_PI
    self.theta1 = theta1
    self.theta2 = theta1 + theta_arc
    self.theta_arc = theta_arc
    self.center_point = center_point
    return True