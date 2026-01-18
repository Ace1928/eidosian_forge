from direct.showbase.ShowBase import ShowBase
from panda3d.core import PointLight, AmbientLight, Vec4, Vec3, DirectionalLight
from panda3d.core import CollisionTraverser, CollisionNode
from panda3d.core import CollisionHandlerPusher, CollisionSphere
from panda3d.bullet import BulletWorld, BulletBoxShape, BulletRigidBodyNode
from panda3d.bullet import BulletSphereShape
from panda3d.core import MouseWatcher, ModifierButtons, PGMouseWatcherBackground
from panda3d.core import WindowProperties
import random
import logging
class GameEnvironmentInitializer(ShowBase):

    def __init__(self):
        super().__init__()
        logging.debug('GameEnvironmentInitializer: Superclass initialization complete.')
        self.configurePhysicsWorld()
        self.initializeCollisionHandlingSystem()
        self.constructEnvironmentalElements()
        self.configureWindowCameraAndBackground()
        self.initializeLightingSystem()
        self.scheduleRegularUpdates()
        self.setFrameRate()
        self.enablePlayerMovement()
        self.bindMouseToCamera()

    def configureWindowCameraAndBackground(self):
        properties = WindowProperties()
        properties.setBackgroundColor(0.1, 0.1, 0.1)
        self.win.requestProperties(properties)
        self.disableMouse()
        self.camera.reparent_to(self.playerNodePath)
        self.camera.set_pos(0, -10, 3)
        self.camera.look_at(self.playerNodePath)
        logging.info('GameEnvironmentInitializer: Window, camera, and background color configured.')

    def initializeLightingSystem(self):
        self.configureAmbientLight()
        self.configurePointLight()
        self.configureDirectionalLight()

    def configureAmbientLight(self):
        ambientLight = AmbientLight('ambient_light')
        ambientLight.setColor(Vec4(0.2, 0.2, 0.2, 1))
        ambientLightNode = self.render.attachNewNode(ambientLight)
        self.render.setLight(ambientLightNode)
        logging.info('GameEnvironmentInitializer: Ambient light configured.')

    def configurePointLight(self):
        pointLight = PointLight('point_light')
        pointLight.setColor(Vec4(1, 1, 1, 1))
        pointLightNode = self.render.attachNewNode(pointLight)
        pointLightNode.set_pos(5, -15, 10)
        self.render.setLight(pointLightNode)
        logging.info('GameEnvironmentInitializer: Point light configured.')

    def configureDirectionalLight(self):
        directionalLight = DirectionalLight('directional_light')
        directionalLight.setColor(Vec4(0.8, 0.8, 0.8, 1))
        directionalLightNode = self.render.attachNewNode(directionalLight)
        directionalLightNode.setHpr(0, -60, 0)
        self.render.setLight(directionalLightNode)
        logging.info('GameEnvironmentInitializer: Directional light configured.')

    def configurePhysicsWorld(self):
        self.world = BulletWorld()
        self.world.setGravity(Vec3(0, 0, -9.81))
        logging.info('GameEnvironmentInitializer: Physics world configured with gravity.')

    def initializeCollisionHandlingSystem(self):
        self.traverser = CollisionTraverser()
        self.pusher = CollisionHandlerPusher()
        logging.info('GameEnvironmentInitializer: Collision handling system initialized.')

    def constructEnvironmentalElements(self):
        self.createGround()
        self.createPlayer()
        self.createObstacles()

    def createGround(self):
        shape = BulletBoxShape(Vec3(20, 20, 1))
        body = BulletRigidBodyNode('Ground')
        body.addShape(shape)
        nodePath = self.render.attachNewNode(body)
        nodePath.set_pos(0, 0, -2)
        nodePath.set_color(0.3, 0.3, 0.3, 1)
        self.world.attachRigidBody(body)
        logging.debug('GameEnvironmentInitializer: Ground element created and positioned.')

    def createPlayer(self):
        shape = BulletSphereShape(1)
        body = BulletRigidBodyNode('Player')
        body.setMass(1.0)
        body.addShape(shape)
        self.playerNodePath = self.render.attachNewNode(body)
        self.playerNodePath.set_pos(0, 0, 2)
        self.playerNodePath.set_color(1, 1, 1, 1)
        self.world.attachRigidBody(body)
        self.definePlayerCollisionSphere()

    def definePlayerCollisionSphere(self):
        collisionNode = CollisionNode('player')
        collisionNode.addSolid(CollisionSphere(0, 0, 0, 1))
        collisionNodePath = self.playerNodePath.attachNewNode(collisionNode)
        self.traverser.addCollider(collisionNodePath, self.pusher)
        logging.debug('GameEnvironmentInitializer: Player collision sphere defined.')

    def createObstacles(self):
        for _ in range(20):
            x, y, z = (random.uniform(-18, 18), random.uniform(-18, 18), 0)
            shape = BulletBoxShape(Vec3(1, 1, 1))
            body = BulletRigidBodyNode('Obstacle')
            body.addShape(shape)
            nodePath = self.render.attachNewNode(body)
            nodePath.set_pos(x, y, z)
            self.world.attachRigidBody(body)
        logging.debug('GameEnvironmentInitializer: Obstacles created and positioned.')

    def scheduleRegularUpdates(self):
        self.taskMgr.add(self.updatePhysicsAndLogging, 'update')
        logging.info('GameEnvironmentInitializer: Regular updates scheduled.')

    def setFrameRate(self):
        globalClock.setFrameRate(120)
        logging.info('GameEnvironmentInitializer: Frame rate set to 120 FPS.')

    def enablePlayerMovement(self):
        self.accept('arrow_up', self.applyMovement, [Vec3(0, 100, 0)])
        self.accept('arrow_down', self.applyMovement, [Vec3(0, -100, 0)])
        self.accept('arrow_left', self.applyMovement, [Vec3(-100, 0, 0)])
        self.accept('arrow_right', self.applyMovement, [Vec3(100, 0, 0)])

    def applyMovement(self, force):
        body = self.playerNodePath.node()
        body.setActive(True)
        body.applyCentralForce(force)

    def bindMouseToCamera(self):
        self.taskMgr.add(self.controlCamera, 'controlCamera')

    def controlCamera(self, task):
        if self.mouseWatcherNode.hasMouse():
            mpos = self.mouseWatcherNode.getMouse()
            self.camera.setP(self.camera.getP() - mpos.getY() * 50)
            self.camera.setH(self.camera.getH() - mpos.getX() * 50)
        return task.cont

    def movePlayer(self, direction):
        self.playerNodePath.setPos(self.playerNodePath.getPos() + direction)
        logging.debug(f'GameEnvironmentInitializer: Player moved to {self.playerNodePath.getPos()}')

    def updatePhysicsAndLogging(self, task):
        deltaTime = globalClock.get_dt()
        self.world.doPhysics(deltaTime)
        playerPosition = self.playerNodePath.get_pos()
        logging.debug(f'GameEnvironmentInitializer: Physics updated for deltaTime: {deltaTime}, Player position: {playerPosition}')
        return task.cont